"""Actual MuJoCo-Warp execution with fresh-scene replay and no auto-reset."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from physics_prefix import Motion,Execution
from codec262 import matrix_quat,matrix_rotvec,derivative
from mjlab_task import make_config,convert_motion,PhyGileMjlabEnv,initialize_relative
from frozen_policy import FrozenPhyGile

SCHEMA='g1-root13-q29-dq29-env_origin_relative-v1'

class ActualMjlab:
    def __init__(self,assets,checkpoint,codec,seed,output,contact_profile):
        self.codec=codec;self.assets=Path(assets);self.output=Path(output)
        self.output.mkdir();self.contract=json.loads((self.assets/'contract.json').read_text(encoding='utf-8'))
        c=self.contract;cfg=make_config(c,self.output/'converted_motion.npz',500,seed,contact_profile)
        entity=cfg.scene.entities['robot'].build()
        convert_motion(self.assets/'test_000.npz',self.output/'converted_motion.npz',entity.joint_names,entity.body_names)
        self.env=PhyGileMjlabEnv(cfg,c,device='cpu',render_mode=None)
        self.robot=self.env.scene['robot'];self.motion=self.env.command_manager.get_term('motion')
        self.env.reset(seed=seed);initialize_relative(self.motion)
        self.policy=FrozenPhyGile(checkpoint,c,self.assets/'encoder_actor_critic.py','cpu')
        self.natural=[c['joint_names'].index(n) for n in self.robot.joint_names]
        self.inverse=self.motion.policy_joint_ids;self.body_names=tuple(c['tracked_body_names'])
        self.calls=[]

    def current(self):
        r=self.robot.data
        root=torch.cat([r.root_link_pos_w-self.env.scene.env_origins,r.root_link_quat_w,r.root_link_lin_vel_w,r.root_link_ang_vel_w],-1)
        state=torch.cat([root,r.joint_pos[:,self.inverse],r.joint_vel[:,self.inverse]],-1).cpu().numpy()
        body=(self.motion.robot_body_pos_w-self.env.scene.env_origins[:,None,:]).cpu().numpy()
        return Motion(state,body,50.,SCHEMA,self.body_names)

    def install(self,candidate):
        s=candidate.states;positions,rotations=self.codec.fk(s[:,:3],s[:,3:7],s[:,13:42])
        names=self.robot.body_names
        pos=np.stack([positions[n] for n in names],1);rot=np.stack([rotations[n] for n in names],1)
        angular=matrix_rotvec(rot[1:]@rot[:-1].transpose(0,1,3,2))*50
        arrays={'joint_pos':s[:,13:42][:,self.natural],'joint_vel':s[:,42:71][:,self.natural],
                '_body_pos_w':pos,'_body_quat_w':matrix_quat(rot),'_body_lin_vel_w':derivative(pos,50.),
                '_body_ang_vel_w':np.concatenate([angular,angular[-1:]],0)}
        for name,value in arrays.items():setattr(self.motion.motion,name,torch.tensor(np.array(value,copy=True),dtype=torch.float32))
        # Mjlab's loader caches these selected arrays; refresh both representations.
        for name in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
            setattr(self.motion.motion,name,getattr(self.motion.motion,'_'+name)[:,self.motion.motion._body_indexes])
        self.motion.motion.time_step_total=len(candidate)

    @torch.inference_mode()
    def rollout(self,candidate,prefix_frames):
        start=self.current().states[0].copy()
        np.testing.assert_allclose(start,candidate.states[prefix_frames-1],atol=2e-6,rtol=0)
        self.install(candidate);states=[];bodies=[];flags=[];actions=[];references=[]
        for frame in range(prefix_frames,len(candidate)):
            self.motion.time_steps.fill_(frame);initialize_relative(self.motion)
            obs=self.env.observation_manager.compute(update_history=False)['actor']
            action=self.policy(obs)[:,self.natural];self.env.action_manager.process_action(action)
            for _ in range(self.env.cfg.decimation):
                self.env._sim_step_counter+=1;self.env.action_manager.apply_action()
                self.env.scene.write_data_to_sim();self.env.sim.step();self.env.scene.update(dt=self.env.physics_dt)
            self.env.episode_length_buf+=1;self.env.common_step_counter+=1
            self.env.sim.forward();self.env.termination_manager.compute()
            done=bool(self.env.termination_manager.terminated[0] or self.env.termination_manager.time_outs[0])
            current=self.current();states.append(current.states[0]);bodies.append(current.body_positions[0]);flags.append(done)
            actions.append(action[0,self.inverse].cpu().numpy())
            references.append((self.motion.body_pos_w[0]-self.env.scene.env_origins[0]).cpu().numpy())
            if done:break
        actual=Motion(np.asarray(states),np.asarray(bodies),50.,SCHEMA,self.body_names)
        np.testing.assert_allclose(references,candidate.body_positions[prefix_frames:prefix_frames+len(actual)],atol=3e-5,rtol=0)
        self.calls.append({'states':actual.states,'body_positions':actual.body_positions,'terminated':np.asarray(flags,dtype=bool),
                           'actions':np.asarray(actions),'reference_body_positions':np.asarray(references),'start_state':start})
        return Execution(actual,np.asarray(flags,dtype=bool),candidate.states[prefix_frames-1].copy(),candidate.digest())

    def close(self):self.env.close()

class TransactionalMjlab:
    def __init__(self,assets,checkpoint,codec,seed,output,contact_profile='matched_sliding_friction'):
        self.args=(assets,checkpoint,codec,seed);self.output=Path(output);self.output.mkdir()
        self.contact_profile=contact_profile;self.ledger=[];self.calls=[];self.invocations=[]
        runtime=self._new();self._current=runtime.current();runtime.close()
    def _new(self):
        out=self.output/('scene_'+str(len(self.invocations)).zfill(3))
        runtime=ActualMjlab(*self.args,out,self.contact_profile)
        self.invocations.append({'directory':str(out),'fresh_scene':True,'replayed_windows':[]})
        return runtime
    def current(self):return self._current
    def snapshot(self):return {'ledger':list(self.ledger),'current':self._current}
    def restore(self,state):self.ledger=list(state['ledger']);self._current=state['current']
    def rollout(self,candidate,prefix_frames):
        runtime=self._new();replays=[]
        try:
            for previous,n,expected in self.ledger:
                actual=runtime.rollout(previous,n)
                if len(actual.continuation)!=len(expected):raise AssertionError('Replay horizon changed')
                delta=float(np.abs(actual.continuation.states-expected.states).max())
                if delta>2e-6 or actual.terminated.any():raise AssertionError('Committed history replay diverged')
                replays.append({'candidate_sha256':previous.digest(),'max_state_difference':delta})
            execution=runtime.rollout(candidate,prefix_frames)
            self.invocations[-1]['replayed_windows']=replays
            self.calls.append(runtime.calls[-1]);self.ledger.append((candidate,prefix_frames,execution.continuation))
            self._current=execution.continuation.slice(-1)
            return execution
        finally:runtime.close()
    def close(self):pass
