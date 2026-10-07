"""Frozen actual Isaac control confirmation. No model/normalizer updates.

Evaluation allocation and first-episode censoring are fixed before any score.
Native scene poses are saved for separate rendering; telemetry never waits.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['diagnostic','formal'],required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--num_envs',type=int,default=2048)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--reference-manifest',type=Path,required=True)
p.add_argument('--monitor-prefix',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();sys.argv=[sys.argv[0]]
launcher=AppLauncher(a);app=launcher.app

import faulthandler,hashlib,json,time
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder
from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.modules import EmpiricalNormalization
from r3_i1_training_telemetry import SampledTrainingTelemetry
from r3_world_ppo_core import WorldModel,HistoryBuffer,AugmentedActorCritic,SCHEMA_HASH
from r3_world_ppo_residual_core import ResidualWorldModel
from r3_world_ppo_training_core_20261007 import feature_observation,tensor_hash

RUNNING_ENV=None
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,data):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def main():
 global RUNNING_ENV
 assert not a.output.exists();a.output.mkdir(parents=True)
 if a.mode=='formal':assert a.steps==500 and a.num_envs==2048 and a.seed in [42,43,44]
 else:assert a.steps==120 and a.num_envs==64 and a.seed==42
 faulthandler.enable();faulthandler.dump_traceback_later(240,repeat=True)
 torch.manual_seed(a.seed);np.random.seed(a.seed);started=time.time()
 c=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
 assert c['schema_hash']==SCHEMA_HASH and c['no_extra_IK']
 assert c['completed_updates']==(500 if a.mode=='formal' else 2)
 assert c['diagnostic_only']==(a.mode!='formal')
 arm=c['arm'];original=torch.load(ORIGINAL,map_location=a.device,weights_only=False)
 template=WorldModel().to(a.device)
 policy=AugmentedActorCritic(arm,original['model_state_dict'],template.encoder.state_dict()).to(a.device)
 policy.load_state_dict(c['policy']);policy.eval()
 norm=EmpiricalNormalization(626).to(a.device);norm.load_state_dict(c['actor_obs_normalizer']);norm.eval()
 world=None
 if arm.startswith('W-'):
  state=c['world_model'];world=ResidualWorldModel({k:state['ridge_'+str(k)] for k in [1,5,10]}).to(a.device)
  world.load_state_dict(state);world.eval()
 for module in [policy,norm,world]:
  if module is not None:
   for parameter in module.parameters():parameter.requires_grad_(False)
 before={name:tensor_hash(module.state_dict()) for name,module in [('policy',policy),('norm',norm),('world',world)] if module is not None}
 manifest=json.loads(a.reference_manifest.read_text());assert manifest['selected_count']==11
 reference_dir=Path(manifest['reference_dir'])
 for row in manifest['selected']:assert sha(reference_dir/row['file'])==row['sha256']
 cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=a.num_envs)
 cfg.seed=a.seed;cfg.ref_motions_path=str(reference_dir);cfg.observations.policy.enable_corruption=False
 # Evaluation only: replace initial random reference assignment with the same
 # balanced table for every arm. Original events handle all subsequent resets.
 allocation=np.arange(a.num_envs,dtype=np.int64)
 reference=allocation%11;within=allocation//11;domain=within%2
 direction=np.where((within//2)%2==0,1.,-1.)
 original_reset=cfg.events.reset_robot_joints.func
 assigned=torch.zeros(a.num_envs,dtype=torch.bool,device=a.device)
 fixedref=torch.as_tensor(reference,device=a.device)
 def balanced_initial_joints(env,env_ids,asset_cfg):
  original_reset(env,env_ids,asset_cfg)
  if not hasattr(env,'motion_lib') or not hasattr(env,'start_motion_times'):return
  ids=env_ids[~assigned[env_ids]]
  if not len(ids):return
  env.motion_ids[ids]=fixedref[ids];env.start_motion_times[ids]=0.
  refstate=env.motion_lib.get_motion_state(env.motion_ids[ids],env.start_motion_times[ids])
  robot=env.scene[asset_cfg.name]
  q=torch.zeros_like(robot.data.default_joint_pos[ids]);v=robot.data.default_joint_vel[ids].clone()
  q[:,asset_cfg.joint_ids]=refstate['dof_pos']
  robot.write_joint_state_to_sim(q,v,env_ids=ids);assigned[ids]=True
 cfg.events.reset_robot_joints.func=balanced_initial_joints
 env=SampledTrainingTelemetry(gym.make('Isaac-Motion-Tracking-Kick-v0',cfg=cfg),'127.0.0.1:40123')
 RUNNING_ENV=env
 print('EVAL_INIT_ENV_CREATED',flush=True)
 raw=env.unwrapped;robot=raw.scene['robot'];lib=raw.motion_lib
 print('EVAL_INIT_BEFORE_RESET',flush=True)
 obs,_=env.reset(seed=a.seed);obs=obs['policy']
 print('EVAL_INIT_RESET_RETURNED',flush=True)
 assert assigned.all() and torch.equal(raw.motion_ids,fixedref)
 assert len(lib._motion_data_list)==11 and torch.count_nonzero(raw.start_motion_times)==0
 print('EVAL_INIT_ALLOCATION_PASSED',flush=True)
 loaded=[Path(f).name for f in lib._motion_data_load]
 print('EVAL_INIT_REFERENCE_ORDER',json.dumps({'loaded':loaded,'manifest':[row['file'] for row in manifest['selected']]}),flush=True)
 assert loaded==[row['file'] for row in manifest['selected']], 'Reference order must match fixed manifest'
 expected=np.stack([np.asarray(d['joints'])[:200] for d in lib._motion_data_list])
 ref_error=float(np.max(np.abs(lib.dof_pos.cpu().numpy()-expected)));assert ref_error<1e-6
 jid=[robot.joint_names.index(x) for x in JointNamesOrder]
 torso=robot.body_names.index('torso_link');history=HistoryBuffer(a.num_envs,a.device);history.append(obs)
 first_alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
 first_failure=torch.zeros_like(first_alive);first_timeout=torch.zeros_like(first_alive);first_success=torch.zeros_like(first_alive)
 first_done_step=torch.full((a.num_envs,),a.steps,device=a.device,dtype=torch.int32)
 termnames=list(raw.termination_manager.active_terms)
 assert set(termnames)=={'time_out','torso_below_threshold','torso_angle_below_threshold'},termnames
 metrics=np.empty((a.steps,a.num_envs),np.float32);unpenalized=np.empty_like(metrics)
 termtrace=np.zeros((a.steps,a.num_envs),bool);timeouttrace=np.zeros_like(termtrace)
 reasontrace=np.zeros((a.steps,a.num_envs,2),bool)
 alive_trace=np.zeros_like(termtrace);rewards=np.zeros_like(metrics)
 force_audit=[];scene=[]
 # One clean and three pushed actual environments across fixed references;
 # trace is native simulation data, not a rendered video or checkpoint replay.
 selected_env=np.array([0,11,33,12])
 root_initial=robot.data.root_state_w.clone();joint_initial=robot.data.joint_pos.clone()
 atomic(ROOT/'artifacts/monitoring/r3_active_dreamcontrol_training.json',{
  'task_id':'Isaac-Motion-Tracking-Kick-v0','stage':'evaluation','is_formal_training':False,
  'source':'actual_isaaclab_simulation','output':str(a.output),'arm':arm,'seed':a.seed,
  'queue_state':str(a.monitor_prefix.with_suffix('.state')),'log':str(a.monitor_prefix.with_suffix('.log'))})
 try:
  with torch.inference_mode():
   for step in range(a.steps):
    ref=lib.get_motion_state(raw.motion_ids,raw.episode_length_buf*raw.step_dt+raw.start_motion_times)
    error=(robot.data.joint_pos[:,jid]-ref['dof_pos']).square().mean(-1).sqrt()
    unpenalized[step]=error.cpu().numpy();metrics[step]=torch.where(first_alive,error,2.).cpu().numpy()
    alive_trace[step]=first_alive.cpu().numpy()
    scene.append({'root':robot.data.root_state_w[selected_env].cpu().numpy().copy(),
      'joint':robot.data.joint_pos[selected_env].cpu().numpy().copy(),
      'body_pose':torch.cat([robot.data.body_pos_w[selected_env],robot.data.body_quat_w[selected_env]],-1).cpu().numpy().copy()})
    worldforce=torch.zeros(a.num_envs,3,device=a.device)
    if 50<=step<60:worldforce[:,1]=80.*torch.as_tensor(domain*direction,device=a.device)
    local=quat_apply_inverse(robot.data.body_quat_w[:,torso],worldforce)[:,None,:]
    robot.set_external_force_and_torque(local,torch.zeros_like(local),body_ids=[torso])
    ferr=float((robot._external_force_b[:,torso]-local[:,0]).abs().max());assert ferr<1e-6
    if 50<=step<60:force_audit.append({'step':step,'force_buffer_error':ferr,'push_count':int((domain==1).sum()),'N':80})
    value=feature_observation(arm,norm,obs,history.value,world);action=policy.act_inference(value)
    success_before=raw.n_successes.clone()
    newobs,reward,terminated,truncated,info=env.step(action)
    done=terminated|truncated
    reasons=torch.stack([raw.termination_manager.get_term(name) for name in
      ['torso_below_threshold','torso_angle_below_threshold']],-1)
    assert torch.equal(reasons.any(-1),terminated)
    reasontrace[step]=reasons.cpu().numpy()
    first_success |= first_alive & (raw.n_successes>success_before)
    first_failure |= first_alive & terminated
    first_timeout |= first_alive & truncated
    first_done_step[first_alive&done]=step
    termtrace[step]=terminated.cpu().numpy();timeouttrace[step]=truncated.cpu().numpy()
    first_alive &=~done;obs=newobs['policy'];history.append(obs,done)
    rewards[step]=reward.cpu().numpy()
    assert torch.isfinite(obs).all() and torch.isfinite(action).all() and torch.isfinite(reward).all()
    if step%25==0:print('WORLD_PPO_EVAL_STEP',step,a.steps,flush=True)
   after={name:tensor_hash(module.state_dict()) for name,module in [('policy',policy),('norm',norm),('world',world)] if module is not None}
   assert before==after
  recovery=metrics[60:110].sum(0)*.02
  rows=[];failure=first_failure.cpu().numpy();success=first_success.cpu().numpy()
  for r in range(11):
   ix=reference==r;clean=ix&(domain==0);push=ix&(domain==1)
   assert clean.sum()>0 and push.sum()>0
   rows.append({'reference_index':r,'reference_file':loaded[r],'clean_count':int(clean.sum()),'push_count':int(push.sum()),
    'push_positive_count':int((push&(direction>0)).sum()),'push_negative_count':int((push&(direction<0)).sum()),
    'recovery_joint_RMSE_AUC_rad_s':float(recovery[push].mean()),
    'clean_recovery_RMSE_rad':float(metrics[60:110,clean].mean()),
    'clean_500_RMSE_rad':float(metrics[:,clean].mean()),
    'first_episode_success':float(success[ix].mean()),'first_episode_failure':float(failure[ix].mean()),
    'first_episode_timeout':float(first_timeout.cpu().numpy()[ix].mean()),
    'survival_seconds':float(((first_done_step.cpu().numpy()[ix]+1).clip(max=a.steps)*.02).mean())})
  np.savez_compressed(a.output/'actual_first_episode_trace.npz',joint_rmse_with_failure_penalty=metrics,
    joint_rmse_unpenalized=unpenalized,first_alive=alive_trace,terminated=termtrace,truncated=timeouttrace,
    failure_termination_reasons=reasontrace,
    reward=rewards,reference=reference,domain=domain,direction=direction,
    first_success=success,first_failure=failure,first_done_step=first_done_step.cpu().numpy(),
    initial_root_state=root_initial.cpu().numpy(),initial_joint_state=joint_initial.cpu().numpy(),
    selected_env=selected_env,body_pose=np.stack([v['body_pose'] for v in scene]),
    root_state=np.stack([v['root'] for v in scene]),joint_state=np.stack([v['joint'] for v in scene]))
  summary={'status':'COMPLETE','mode':a.mode,'arm':arm,'seed':a.seed,'num_envs':a.num_envs,'steps':a.steps,
    'actual_evaluation_transitions':a.num_envs*a.steps,'PPO_updates':0,'aux_updates':0,'normalizer_updates':0,
    'checkpoint_sha256':sha(a.checkpoint),'checkpoint_completed_updates':c['completed_updates'],
    'reference_manifest_sha256':sha(a.reference_manifest),'reference_maxerror':ref_error,
    'model_hashes_unchanged':before==after,'deterministic_actor':True,'corruption':False,
    'initial_phase':0,'rows':rows,'termination_terms':termnames,'force_audit':force_audit,
    'official_any_reset_success_rate':float((raw.n_successes>0).float().mean()),
    'first_episode_macro_success':float(np.mean([r['first_episode_success'] for r in rows])),
    'first_episode_macro_failure':float(np.mean([r['first_episode_failure'] for r in rows])),
    'recovery_macro_joint_RMSE_AUC':float(np.mean([r['recovery_joint_RMSE_AUC_rad_s'] for r in rows])),
    'clean_macro_recovery_RMSE':float(np.mean([r['clean_recovery_RMSE_rad'] for r in rows])),
    'native_scene_trace_sha256':sha(a.output/'actual_first_episode_trace.npz'),
    'evaluation_video_verified':False,'trace_is_not_video':True,'no_extra_IK':True,
    'innovation_validated':False,'wall_s':time.time()-started}
  atomic(a.output/'summary.json',summary);print('WORLD_PPO_CONTROL_EVALUATION_COMPLETE',flush=True)
 finally:env.close();faulthandler.cancel_dump_traceback_later()
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  print('EVAL_ORIGINAL_EXCEPTION_BEFORE_CLEANUP',flush=True)
  traceback.print_exc();sys.stderr.flush()
  if RUNNING_ENV is not None:
   try:RUNNING_ENV.close()
   except BaseException:traceback.print_exc();sys.stderr.flush()
  raise
 finally:app.close()
