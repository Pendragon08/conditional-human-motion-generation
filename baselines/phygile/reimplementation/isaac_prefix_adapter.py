"""Actual Isaac Lab 2.1 adapter, without automatic reset during a proposal.

This runtime only steps forward. Direct state-write rollback failed a measured
repeatability check and is disabled. ReplayPrefixAdapter recreates the seeded
scene and replays accepted history to restore solver and controller evolution.
"""
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationEnvCfg
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
from physics_prefix import Motion,Execution

STATE_SCHEMA='g1-root13-q29-dq29-env_origin_relative-v1'


class IsaacPrefixAdapter:
    def __init__(self,checkpoint,motion_file,device,seed,codec):
        self.codec=codec;self.device=device;self.seed=seed;self.calls=[]
        cfg=G1FlatWoStateEstimationEnvCfg();cfg.scene.num_envs=1;cfg.seed=seed;cfg.sim.device=device
        cfg.sim.physx.enable_enhanced_determinism=True
        cfg.commands.motion.motion_dir=None;cfg.commands.motion.motion_file=str(motion_file)
        cfg.commands.motion.curriculum_enabled=False;cfg.commands.motion.random_static_prob=0
        cfg.commands.motion.debug_vis=False;cfg.scene.contact_forces.debug_vis=False
        cfg.events.push_robot=None;cfg.observations.policy.enable_corruption=False
        cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={}
        cfg.commands.motion.joint_position_range=(0.,0.)
        cfg.episode_length_s=10.
        self.env=gym.make('Tracking-Flat-G1-Wo-State-Estimation-v0',cfg=cfg)
        self.base=self.env.unwrapped;self.robot=self.base.scene['robot']
        self.motion=self.base.command_manager.get_term('motion')
        self.motion._adaptive_sampling=lambda ids:self.motion.time_steps.__setitem__(ids,0)
        wrapped=RslRlVecEnvWrapper(self.env)
        agent=G1FlatPPORunnerCfg();agent.device=device;agent.seed=seed;agent.logger='tensorboard'
        self.runner=MotionOnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=device,
            upload_models_to_wandb=False)
        self.runner.load(str(checkpoint),load_optimizer=False)
        self.policy=self.runner.get_inference_policy(device=device)
        self.runner.alg.policy.eval()
        self.base.reset(seed=seed)
        print('CLOSED_LOOP_STAGE initial_reset_ready',flush=True)
        assert tuple(self.robot.joint_names)==codec.joint_names
        assert tuple(self.robot.body_names)==codec.body_names
        assert abs(self.base.step_dt-.02)<1e-8
        self.body_names=tuple(self.motion.cfg.body_names)
        self.body_indices=[codec.body_names.index(n) for n in self.body_names]
        # Exclude history-bearing tasks until they have an explicit restoration
        # implementation. Current pinned G1 task has no observation history.
        for group in self.base.observation_manager._group_obs_term_cfgs.values():
            for term in group:
                if getattr(term,'history_length',0)>0:raise ValueError('Observation history not supported by this adapter')

    def current(self):
        root=self.robot.data.root_state_w.clone();root[:,:3]-=self.base.scene.env_origins
        states=torch.cat([root,self.robot.data.joint_pos,self.robot.data.joint_vel],-1).cpu().numpy()
        bodies=(self.motion.robot_body_pos_w-self.base.scene.env_origins[:,None,:]).cpu().numpy()
        return Motion(states,bodies,50.,STATE_SCHEMA,self.body_names)

    def _set_relative(self):
        m=self.motion
        anchor=m.anchor_pos_w[:,None,:].repeat(1,len(m.cfg.body_names),1)
        anchor_q=m.anchor_quat_w[:,None,:].repeat(1,len(m.cfg.body_names),1)
        robot_q=m.robot_anchor_quat_w[:,None,:].repeat(1,len(m.cfg.body_names),1)
        position=m.robot_anchor_pos_w[:,None,:].repeat(1,len(m.cfg.body_names),1)
        position[...,2]=anchor[...,2];rotation=yaw_quat(quat_mul(robot_q,quat_inv(anchor_q)))
        m.body_quat_relative_w=quat_mul(rotation,m.body_quat_w)
        m.body_pos_relative_w=position+quat_apply(rotation,m.body_pos_w-anchor)

    def install_candidate(self,candidate):
        s=candidate.states
        # Build every full-body reference with FK from the candidate's joint/root
        # state; never substitute observed robot positions for desired targets.
        positions,rotations=self.codec.fk(s[:,:3],s[:,3:7],s[:,13:42])
        from codec262 import matrix_quat,derivative
        pos=np.stack([positions[n] for n in self.codec.body_names],1)
        rot=np.stack([rotations[n] for n in self.codec.body_names],1)
        source={'joint_pos':s[:,13:42],'joint_vel':s[:,42:71],
            '_body_pos_w':pos,'_body_quat_w':matrix_quat(rot),
            '_body_lin_vel_w':derivative(pos,50.),'_body_ang_vel_w':np.zeros_like(pos)}
        # Angular velocities are relevant to reward/terminations and must not be
        # silently zeroed when a candidate turns.
        from codec262 import matrix_rotvec
        angular=matrix_rotvec(rot[1:]@rot[:-1].transpose(0,1,3,2))*50
        source['_body_ang_vel_w']=np.concatenate([angular,angular[-1:]],0)
        for name,value in source.items():
            setattr(self.motion.motion,name,torch.tensor(np.array(value,copy=True),dtype=torch.float32,device=self.device))
        self.motion.motion.time_step_total=len(candidate)
        self.motion.motion.fps=np.array([50.])

    def snapshot(self):
        raise RuntimeError('Direct state-write rollback is unverified; use ReplayPrefixAdapter')

    def restore(self,snapshot):
        raise RuntimeError('Direct state-write rollback is unverified; use ReplayPrefixAdapter')

    @torch.inference_mode()
    def rollout(self,candidate,prefix_frames):
        start=self.current().states[0].copy()
        np.testing.assert_allclose(start,candidate.states[prefix_frames-1],atol=2e-6,rtol=0)
        self.install_candidate(candidate)
        states=[];bodies=[];done=[];references=[];actions=[];timeouts=[]
        for frame in range(prefix_frames,len(candidate)):
            self.motion.time_steps.fill_(frame)
            self._set_relative()
            observations=self.base.observation_manager.compute()['policy']
            action=self.policy(observations)
            if not bool(torch.isfinite(action).all()):raise FloatingPointError('Nonfinite controller action')
            self.base.action_manager.process_action(action)
            for _ in range(self.base.cfg.decimation):
                self.base._sim_step_counter+=1
                self.base.action_manager.apply_action()
                self.base.scene.write_data_to_sim()
                self.base.sim.step(render=False)
                self.base.scene.update(dt=self.base.physics_dt)
            self.base.episode_length_buf+=1;self.base.common_step_counter+=1
            # Critical ordering: physics -> failure detection -> capture. There
            # is no _reset_idx, command resampling or extra frame advancement.
            self.base.termination_manager.compute()
            failed=bool(self.base.termination_manager.terminated[0])
            timeout=bool(self.base.termination_manager.time_outs[0])
            actual=self.current()
            states.append(actual.states[0]);bodies.append(actual.body_positions[0]);done.append(failed or timeout)
            references.append((self.motion.body_pos_w[0]-self.base.scene.env_origins[0]).cpu().numpy())
            actions.append(action[0].cpu().numpy());timeouts.append(timeout)
            if failed or timeout:break
        actual=Motion(np.asarray(states),np.asarray(bodies),50.,STATE_SCHEMA,self.body_names)
        reference=np.asarray(references)
        np.testing.assert_allclose(reference,candidate.body_positions[prefix_frames:prefix_frames+len(actual)],atol=3e-5,rtol=0)
        self.calls.append({'candidate_sha256':candidate.digest(),'states':actual.states.copy(),
            'body_positions':actual.body_positions.copy(),'reference_body_positions':reference,
            'terminated':np.asarray(done,dtype=bool),'timeouts':np.asarray(timeouts,dtype=bool),
            'actions':np.asarray(actions),'start_state':start,'frames':np.arange(prefix_frames,prefix_frames+len(actual))})
        # Bind to the last committed bit pattern only after checking the actual
        # simulator state against it at float32 restoration precision.
        return Execution(actual,np.asarray(done,dtype=bool),candidate.states[prefix_frames-1].copy(),candidate.digest())

    def close(self):
        self.env.close()
