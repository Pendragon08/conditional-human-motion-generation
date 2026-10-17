"""Dataset-referenced simulation evaluation captured BEFORE automatic reset.

Uses upstream termination functions and policy. Metrics match the policy's
current commanded reference (t), after one physics/control interval. This is
a documented pilot protocol, not a claim to reproduce the paper's benchmark.
"""
import argparse, os, pathlib, json, hashlib, sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--motion_file',required=True)
p.add_argument('--output',required=True);p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--task',default='Tracking-Flat-G1-Wo-State-Estimation-v0')
p.add_argument('--seed',type=int,default=2026);p.add_argument('--g1_training_cfg',default='configs/moe_pilot.yaml')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
os.environ['G1_TRAINING_CFG_PATH']=str(pathlib.Path(a.g1_training_cfg).resolve())
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
from guarded_bootstrap import activate
guarded_receipt=activate()
app=AppLauncher(a).app
import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatWoStateEstimationEnvCfg
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
motion_file=pathlib.Path(a.motion_file).resolve();checkpoint=pathlib.Path(a.checkpoint).resolve()
with np.load(motion_file,allow_pickle=False) as d:
 nframes=d['joint_pos'].shape[0];fps=float(d['fps'].reshape(-1)[0])
 joint_names=d['joint_names'].tolist();body_names=d['body_names'].tolist()
cfg=(G1FlatWoStateEstimationEnvCfg() if a.task=='Tracking-Flat-G1-Wo-State-Estimation-v0' else G1FlatEnvCfg());cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device
cfg.commands.motion.motion_dir=None;cfg.commands.motion.motion_file=str(motion_file)
cfg.commands.motion.curriculum_enabled=False;cfg.commands.motion.random_static_prob=0
cfg.commands.motion.debug_vis=False;cfg.scene.contact_forces.debug_vis=False
cfg.episode_length_s=nframes/fps
# Keep training reset perturbations and startup physical randomization. Disable
# interval pushes and observation noise so this pilot measures clip tracking.
cfg.events.push_robot=None
cfg.observations.policy.enable_corruption=False
env=gym.make(a.task,cfg=cfg)
base=env.unwrapped;robot=base.scene['robot'];mt=base.command_manager.get_term('motion')
assert robot.joint_names==joint_names,('joint order mismatch',robot.joint_names,joint_names)
assert robot.body_names==body_names,('body order mismatch',robot.body_names,body_names)
assert abs(base.step_dt-1/fps)<1e-8
def reset_start(ids):mt.time_steps[ids]=0
mt._adaptive_sampling=reset_start
wrapped=RslRlVecEnvWrapper(env)
agent=G1FlatPPORunnerCfg();agent.device=a.device;agent.seed=a.seed;agent.logger='tensorboard'
runner=MotionOnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=a.device,upload_models_to_wandb=False)
runner.load(str(checkpoint),load_optimizer=False)
policy=runner.get_inference_policy(device=a.device)
base.reset(seed=a.seed)
assert bool(torch.all(mt.time_steps==0)),mt.time_steps
# Initialize the relative reference exactly as upstream _update_command does,
# without advancing the frame counter or resampling a trajectory.
anchor=mt.anchor_pos_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
anchor_q=mt.anchor_quat_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
robot_q=mt.robot_anchor_quat_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
pos=mt.robot_anchor_pos_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
pos[...,2]=anchor[...,2];rot=yaw_quat(quat_mul(robot_q,quat_inv(anchor_q)))
mt.body_quat_relative_w=quat_mul(rot,mt.body_quat_w)
mt.body_pos_relative_w=pos+quat_apply(rot,mt.body_pos_w-anchor)
base.obs_buf=base.observation_manager.compute();obs,_=wrapped.get_observations()
# Native 71D feedback is explicitly NOT the paper's 262D generator representation.
initial_root=robot.data.root_state_w.clone()
initial_root[:,:3]-=base.scene.env_origins
initial_states=torch.cat([initial_root,robot.data.joint_pos,robot.data.joint_vel],dim=-1).cpu().numpy()
initial_bodies=(mt.robot_body_pos_w-base.scene.env_origins[:,None,:]).cpu().numpy()
active=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
rows=[];original_compute=base.termination_manager.compute
def capture():
 result=original_compute()
 valid=active.clone()
 ids=torch.where(valid)[0]
 if ids.numel():
  root_state=robot.data.root_state_w[ids].clone()
  root_state[:,:3]-=base.scene.env_origins[ids]
  native_state=torch.cat([root_state,robot.data.joint_pos[ids],robot.data.joint_vel[ids]],dim=-1)
  row={'native_state':native_state.cpu().numpy(),
    'reference_body_pos_local':(mt.body_pos_w[ids]-base.scene.env_origins[ids,None,:]).cpu().numpy(),
    'robot_body_pos_local':(mt.robot_body_pos_w[ids]-base.scene.env_origins[ids,None,:]).cpu().numpy(),
    'env_id':ids.cpu().numpy(),'frame':mt.time_steps[ids].cpu().numpy(),
    'reference_body_pos':mt.body_pos_w[ids].cpu().numpy(),'robot_body_pos':mt.robot_body_pos_w[ids].cpu().numpy(),
    'reference_joint_pos':mt.joint_pos[ids].cpu().numpy(),'robot_joint_pos':mt.robot_joint_pos[ids].cpu().numpy(),
    'reference_joint_vel':mt.joint_vel[ids].cpu().numpy(),'robot_joint_vel':mt.robot_joint_vel[ids].cpu().numpy(),
    'terminated':base.termination_manager.terminated[ids].cpu().numpy(),'timeout':base.termination_manager.time_outs[ids].cpu().numpy()}
  rows.append(row)
 return result
base.termination_manager.compute=capture
with torch.inference_mode():
 for step in range(nframes+2):
  actions=policy(obs)
  assert torch.isfinite(actions).all(),'nonfinite actions'
  obs,_,dones,_=wrapped.step(actions)
  active &= ~dones.bool()
  if not active.any():break
assert not active.any(),'evaluation did not finish every first episode'
raw={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
for k,v in raw.items():assert np.isfinite(v).all(),k
expected=np.arange(nframes)
lengths=[];success=[]
for eid in range(a.num_envs):
 idx=raw['env_id']==eid;frames=raw['frame'][idx]
 assert np.array_equal(frames,np.arange(len(frames))),('frame discontinuity',eid,frames)
 lengths.append(len(frames));success.append(bool(len(frames)==nframes and not raw['terminated'][idx].any()))
mpjpe=np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean(axis=-1)
mpjae=np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean(axis=-1)
mpjve=np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean(axis=-1)
dest=pathlib.Path(a.output);dest.mkdir(parents=True,exist_ok=False)
np.savez_compressed(dest/'raw_pre_reset.npz',**raw)
np.savez_compressed(dest/'initial_state.npz',native_state=initial_states,body_positions=initial_bodies)
feedback_meta={'schema':'g1-root13-q29-dq29-env_origin_relative-v1',
 'root_order':'xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz; world axes',
 'joint_names':joint_names,'body_names':list(mt.cfg.body_names),'fps':fps,
 'timing':'post_step_pre_reset_current_reference',
 'source_type':'dataset_reference_replay_diagnostic; no generated reference',
 'guarded_source':guarded_receipt,
 'initial_state_sha256':hashlib.sha256((dest/'initial_state.npz').read_bytes()).hexdigest(),
 'raw_sha256':hashlib.sha256((dest/'raw_pre_reset.npz').read_bytes()).hexdigest()}
(dest/'feedback_metadata.json').write_text(json.dumps(feedback_meta,indent=2)+'\n')
result={'evaluation_type':'simulation_only','reference_type':'dataset-derived kinematic reference; not model output','protocol':'first episode from frame 0; pre-reset post-physics state versus current command t; training failure terminations retained','checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'motion_file':str(motion_file),'motion_sha256':hashlib.sha256(motion_file.read_bytes()).hexdigest(),'seed':a.seed,'num_envs':a.num_envs,'frames':nframes,'fps':fps,'active_experts':runner.alg.policy._max_active_experts,'episodes_completed':a.num_envs,'successes':sum(success),'full_clip_success_fraction':float(np.mean(success)),'episode_lengths':lengths,'steps_evaluated':len(mpjpe),'metrics':{'mpjpe_m_until_termination':float(mpjpe.mean()),'mpjae_rad_until_termination':float(mpjae.mean()),'mpjve_rad_s_until_termination':float(mpjve.mean())},'note':'Errors are censored at failure; compare together with success and episode lengths. No paper-scale or unseen-task claim.'}
(dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print('PHYGILE_PREFIX_FEEDBACK_CAPTURE',json.dumps(result))
env.close();app.close()
