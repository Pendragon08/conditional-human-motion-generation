"""One live PhysX episode paused at a causal replan boundary. No PPO updates.

Randomized reference blocks, not an exact cross-arm solver clone. The actor,
clock, original target and robot state never reset at the branch. No telemetry
is sent to the existing training viewer. All new outputs are exclusive.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--witness',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();sys.argv=[sys.argv[0]]
launcher=AppLauncher(a);app=launcher.app
import hashlib,json,time,traceback
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.managers import TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import MotionLibBase
from isaaclab.utils.math import quat_apply_inverse,quat_apply,quat_mul
from isaac_utils.rotations import slerp
ROOT=Path(__file__).resolve().parents[1]
POLICY=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/exported/policy.pt')
ENV=None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 global ENV
 conf=json.loads(a.protocol.read_text())
 assert a.seed in conf['seeds'] and not a.output.exists()
 a.output.mkdir(parents=True)
 np.random.seed(a.seed);torch.manual_seed(a.seed)
 refs=conf['references'][:2] if a.witness else conf['references']
 logical=[{'arm':arm,'reference':rid,'domain':domain} for rid in refs for domain in conf['domains'] for arm in conf['arms']]
 rng=np.random.default_rng(a.seed)
 trials=[logical[int(i)] for i in rng.permutation(len(logical))]
 for i,t in enumerate(trials):t['env']=i
 n=len(trials);ids=torch.tensor([t['reference'] for t in trials],device=a.device)
 cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=n)
 cfg.seed=a.seed;cfg.observations.policy.enable_corruption=False
 removed=[]
 for name,value in vars(cfg.terminations).items():
  if isinstance(value,TerminationTermCfg):setattr(cfg.terminations,name,None);removed.append(name)
 cfg.episode_length_s=20.0
 ENV=gym.make('Isaac-Motion-Tracking-Kick-v0',cfg=cfg)
 ENV.reset(seed=a.seed);e=ENV.unwrapped;robot=e.scene['robot'];lib=e.motion_lib
 original=lib.get_motion_state
 archived=np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')
 err=np.max(np.abs(lib.dof_pos.cpu().numpy()-archived['dof_pos']))
 assert err<1e-6
 jid=[robot.joint_names.index(x) for x in lib.joint_names]
 bid=[robot.body_names.index(x) for x in ['pelvis','left_ankle_roll_link','right_ankle_roll_link']]
 torso=robot.body_names.index('torso_link')
 policy=torch.jit.load(str(POLICY),map_location=a.device).eval()
 e.motion_ids[:]=ids;e.start_motion_times[:]=0;e.episode_length_buf[:]=0
 state0=original(ids,torch.zeros(n,device=a.device))
 q=torch.zeros_like(robot.data.default_joint_pos);q[:,jid]=state0['dof_pos']
 robot.write_joint_state_to_sim(q,robot.data.default_joint_vel.clone())
 pos=state0['root_pos']+e.scene.env_origins;pos[:,2]+=.15
 robot.write_root_pose_to_sim(torch.cat([pos,state0['root_rot']],-1))
 robot.write_root_velocity_to_sim(torch.zeros(n,6,device=a.device))
 e.action_manager.reset();e.scene.write_data_to_sim();e.sim.forward();e.scene.update(dt=0.)
 assert e.step_dt==conf['control_dt_s']
 frames=np.array(conf['human_context_frames'])
 dt=float(archived['_motion_lengths'][refs[0]])/(int(archived['_motion_num_frames'][refs[0]])-1)
 assert all(np.all(archived[key][refs]==archived[key][refs[0]]) for key in ['_motion_lengths','_motion_num_frames','_motion_dt'])
 context_steps=np.rint((frames+19)*dt/e.step_dt).astype(int)
 branch=int(np.ceil((frames[-1]+19)*dt/e.step_dt));assert max(context_steps)<=branch
 goal=state0['grab_pos']+state0['offsets']
 fixed={x:getattr(lib,x).clone() for x in ['grab_pos','offsets','switch_idxs']}
 candidates={};accepted={};audit={};branch_active=False;alignment=None
 def switched(motion_ids,times):
  raw=original(motion_ids,times)
  if not branch_active:return raw
  assert torch.equal(motion_ids,ids)
  blend=((times-branch*e.step_dt)/conf['blend_s']).clamp(0,1)[:,None]
  for arm,(c,mapping) in candidates.items():
   valid=[(i,idx) for i,idx in mapping.items()]
   if not valid:continue
   rows=torch.tensor([i for i,_ in valid],device=a.device)
   cids=torch.tensor([idx for _,idx in valid],device=a.device)
   v=c.get_motion_state(cids,times[rows]);shift=raw['offsets'][rows]-v['offsets']
   v['root_pos']+=shift;v['global_keypts']+=shift[:,None,:]
   for field in ['root_pos','dof_pos','local_keypts','global_keypts']:
    weight=blend[rows,:,None] if raw[field].ndim==3 else blend[rows]
    raw[field][rows]=(1-weight)*raw[field][rows]+weight*v[field]
   raw['root_rot'][rows]=slerp(raw['root_rot'][rows],v['root_rot'],blend[rows])
  if alignment is not None:
   rows,rot,pivot,offset=alignment
   newpos=quat_apply(rot,raw['root_pos'][rows]-pivot)+pivot+offset
   key=raw['global_keypts'][rows]
   newkey=quat_apply(rot[:,None,:].expand(-1,key.shape[1],-1).reshape(-1,4),(key-pivot[:,None,:]).reshape(-1,3)).reshape_as(key)+pivot[:,None,:]+offset[:,None,:]
   raw['root_pos'][rows]=(1-blend[rows])*raw['root_pos'][rows]+blend[rows]*newpos
   raw['global_keypts'][rows]=(1-blend[rows,:,None])*key+blend[rows,:,None]*newkey
   raw['root_rot'][rows]=slerp(raw['root_rot'][rows],quat_mul(rot,raw['root_rot'][rows]),blend[rows])
  return raw
 lib.get_motion_state=switched
 keys=['body_xyz','reference_body_xyz','joints','positions','quaternions','velocities','actions','joint_rmse','root_error','fall','goal_error','active_reference_joint_rmse','contact_force','reference_joint_delta']
 records={k:[] for k in keys};prev=None;start=time.time();pause_wall=0.
 push=torch.tensor([t['domain']=='push' for t in trials],device=a.device)
 sensor=e.scene['contact_forces'];cid=[sensor.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
 steps=90 if a.witness else conf['steps']
 for step in range(steps):
  assert torch.equal(e.motion_ids,ids) and torch.all(e.episode_length_buf==step) and torch.all(e.start_motion_times==0)
  if step==branch:
   ref=original(ids,torch.full((n,),step*e.step_dt,device=a.device))
   prefix={k:np.stack(v) for k,v in records.items()}
   prefix['context_body_xyz']=np.concatenate([prefix['body_xyz'],(robot.data.body_pos_w[:,bid]-e.scene.env_origins[:,None,:]).cpu().numpy()[None]],axis=0)
   prefix['context_reference_body_xyz']=np.concatenate([prefix['reference_body_xyz'],ref['global_keypts'][:,[0,7,13]].cpu().numpy()[None]],axis=0)
   np.savez_compressed(a.output/'prefix.npz',**prefix)
   before={'root':robot.data.root_state_w.clone(),'joint':robot.data.joint_pos.clone(),'velocity':robot.data.joint_vel.clone(),'action':e.action_manager.action.clone()}
   metadata={'status':'BRANCH_READY','seed':a.seed,'trials':trials,'references':refs,'branch_step':branch,'context_steps':context_steps.tolist(),'context_human_frames':frames.tolist(),'policy_sha256':sha(POLICY),'original_reference_identity_max_abs_error':float(err),'prefix_sha256':sha(a.output/'prefix.npz'),'protocol_sha256':sha(a.protocol),'feedback_source':'same ongoing actual Isaac Lab episode','no_PPO_updates':True,'no_branch_reset':True,'randomized_physical_slots':True,'exact_cross_arm_solver_clone':False,'world_goal':goal.cpu().tolist(),'original_event_s':(lib.switch_idxs[ids]*lib._motion_dt[ids]).cpu().tolist()}
   atomic(a.output/'branch_ready.json',metadata)
   if not a.witness:
    paused=time.time();ready=a.output/'candidates_ready.json'
    while not ready.exists():
     if (a.output/'abort.json').exists():raise RuntimeError((a.output/'abort.json').read_text())
     if time.time()-paused>conf['max_stage_s']['physics']-900:raise TimeoutError('Paused replan deadline')
     time.sleep(1.)
    pause_wall=time.time()-paused;manifest=json.loads(ready.read_text())
    assert manifest['seed']==a.seed and manifest['branch_ready_sha256']==sha(a.output/'branch_ready.json')
    for arm in conf['generation_arms']:
     folder=Path(manifest['arms'][arm]['folder'])
     assert folder.resolve().is_relative_to((ROOT/'third_party/DreamControl/TrajGen').resolve())
     files=list(folder.glob('*_n.pkl'));accepted[arm]=[]
     if not files:continue
     for file in files:assert sha(file)==manifest['arms'][arm]['files'][file.name]
     c=MotionLibBase(n,a.device,str(folder));c.load_motions()
     file_map={int(Path(f).stem.split('_')[0]):i for i,f in enumerate(c._motion_data_load)}
     pairs=manifest['pairs'];mapping={i:file_map[j] for j,pair in enumerate(pairs) for i,t in enumerate(trials) if t['arm']==arm and t['reference']==pair['reference'] and t['domain']==pair['domain'] and j in file_map}
     accepted[arm]=list(mapping)
     assert torch.all(c._motion_num_frames==lib._motion_num_frames[refs[0]]) and torch.all(c._motion_lengths==lib._motion_lengths[refs[0]])
     limits=robot.data.joint_pos_limits[0,jid]
     excess=torch.maximum(limits[:,0]-c.dof_pos,c.dof_pos-limits[:,1]).clamp_min(0)
     finite=bool(torch.isfinite(c.dof_pos).all() and torch.isfinite(c.transl).all())
     assert finite and float(excess.max())<=1e-4
     audit[arm]={'finite':finite,'max_hard_limit_excess_rad':float(excess.max()),'accepted':len(mapping),'attempted':len(refs)*2}
     candidates[arm]=(c,mapping)
    for name,value in before.items():
     after={'root':robot.data.root_state_w,'joint':robot.data.joint_pos,'velocity':robot.data.joint_vel,'action':e.action_manager.action}[name]
     assert torch.equal(value,after),f'Paused live state changed: {name}'
   rows=torch.tensor([i for i,t in enumerate(trials) if t['arm']=='alignment'],device=a.device)
   def yaw(q):return torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
   angle=yaw(robot.data.root_quat_w[rows])-yaw(ref['root_rot'][rows])
   rot=torch.zeros(len(rows),4,device=a.device);rot[:,0]=torch.cos(angle/2);rot[:,3]=torch.sin(angle/2)
   alignment=(rows,rot,ref['root_pos'][rows].clone(),robot.data.root_pos_w[rows]-e.scene.env_origins[rows]-ref['root_pos'][rows])
   branch_active=not a.witness
  times=torch.full((n,),step*e.step_dt,device=a.device);oref=original(ids,times);active=switched(ids,times)
  obs=e.observation_manager.compute()['policy']
  with torch.inference_mode():action=policy(obs)
  assert obs.shape==(n,626) and action.shape==(n,29) and torch.isfinite(obs).all() and torch.isfinite(action).all()
  qnow=robot.data.joint_pos[:,jid];root=robot.data.root_pos_w-e.scene.env_origins
  body=robot.data.body_pos_w[:,bid]-e.scene.env_origins[:,None,:]
  values={'body_xyz':body,'reference_body_xyz':oref['global_keypts'][:,[0,7,13]],'joints':qnow,'positions':root,'quaternions':robot.data.root_quat_w,'velocities':torch.cat([robot.data.root_vel_w,robot.data.joint_vel[:,jid]],-1),'actions':action,'joint_rmse':(qnow-oref['dof_pos']).square().mean(-1).sqrt(),'root_error':(root-oref['root_pos']).norm(dim=-1),'fall':(root[:,2]<.3)|(robot.data.projected_gravity_b[:,2]>-.5),'goal_error':(body[:,2]-goal).norm(dim=-1),'active_reference_joint_rmse':(qnow-active['dof_pos']).square().mean(-1).sqrt(),'contact_force':sensor.data.net_forces_w[:,cid].norm(dim=-1),'reference_joint_delta':torch.zeros(n,device=a.device) if prev is None else (active['dof_pos']-prev).abs().max(-1).values}
  prev=active['dof_pos'].clone()
  for key,value in values.items():records[key].append(value.detach().cpu().numpy().copy())
  forces=torch.zeros(n,1,3,device=a.device)
  if conf['push_steps'][0]<=step<conf['push_steps'][1]:
   world=torch.zeros(n,3,device=a.device);world[:,1]=push*conf['world_push_N']
   forces[:,0]=quat_apply_inverse(robot.data.body_quat_w[:,torso],world)
  robot.set_external_force_and_torque(forces,torch.zeros_like(forces),body_ids=[torso])
  _,_,term,trunc,_=ENV.step(action);assert not torch.any(term) and not torch.any(trunc)
  if step%25==0:print('FEEDBACK_PAUSED_ACTUAL_STEP',step,'of',steps,'seed',a.seed,flush=True)
 arrays={k:np.stack(v) for k,v in records.items()};np.savez_compressed(a.output/'rollout.npz',**arrays)
 rows=[]
 for i,t in enumerate(trials):
  missing=t['arm'] in conf['generation_arms'] and i not in accepted.get(t['arm'],[]) and not a.witness
  first=np.flatnonzero(arrays['fall'][:,i]);first=int(first[0]) if len(first) else None
  penalty=arrays['joint_rmse'][:,i].copy()
  if first is not None:penalty[first:]=2.
  if missing:penalty[:]=2.
  event=float(lib.switch_idxs[t['reference']]*lib._motion_dt[t['reference']])
  event_mask=np.abs(np.arange(steps)*e.step_dt-event)<=.1
  hits=arrays['goal_error'][event_mask,i]
  rows.append({**t,'retarget_missing':missing,'first_fall_step':first,'failure':missing or first is not None,'post_branch_original_joint_rmse_auc_rad_s':float(penalty[branch:min(branch+90,steps)].sum()*e.step_dt),'post_branch_original_root_error_mean_m':float(arrays['root_error'][branch:,i].mean()),'whole_episode_original_joint_rmse_rad':float(penalty.mean()),'virtual_goal_success':bool(len(hits) and np.any(hits<.1) and not missing and (first is None or first*e.step_dt>event+.1)),'original_goal_event_s':event,'virtual_goal_min_error_m':float(hits.min()) if len(hits) else None,'switch_delta_p95_rad':float(np.percentile(arrays['reference_joint_delta'][branch:min(branch+20,steps),i],95)),'prefix_joint_rmse_mean_rad':float(arrays['joint_rmse'][:branch,i].mean())})
 for name,value in fixed.items():assert torch.equal(getattr(lib,name),value)
 result={'status':'COMPLETE','scope':'WITNESS_ONLY' if a.witness else 'EXPLORATORY_OFFLINE_RANDOMIZED_BLOCK_ABLATION','seed':a.seed,'num_envs':n,'steps':steps,'branch_step':branch,'paused_wall_s':pause_wall,'wall_s':time.time()-start,'frozen_policy_sha256':sha(POLICY),'protocol_sha256':sha(a.protocol),'rollout_sha256':sha(a.output/'rollout.npz'),'rows':rows,'candidate_audits':audit,'PPO_updates':0,'no_extra_IK':True,'same_episode_branch_without_state_reset':True,'future_original_goal_and_event_preserved':True,'randomized_arm_allocation':True,'exact_cross_arm_prefix_identity_claimed':False,'context_is_current_actual_Isaac_episode':True,'real_time_control':False,'physical_entity_contact_evaluated':False,'innovation_validated':False,'auto_resets_disabled':removed}
 atomic(a.output/'summary.json',result)
 print('FEEDBACK_PAUSED_ISAAC_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 ENV.close();ENV=None
try:main()
except BaseException:
 traceback.print_exc();sys.stderr.flush()
 if ENV is not None:
  try:ENV.close()
  except BaseException:traceback.print_exc()
 raise
finally:app.close()
