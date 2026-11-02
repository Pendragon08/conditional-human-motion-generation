"""Strict frozen15k policy replay of all actual LDM100k diagnostic futures.

Uses the already executed strict T-1 coverage/contact/termination mechanics.
Generated-library identity is separate from the policy's original8278 binding.
Independent final-reference position distances supplement the original pass rule.
"""
import argparse,hashlib,json,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
from teacher15000_strict_protocol import load_teacher,authorize_teacher
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
HOME=pathlib.Path('/home/ouyanghao')
STATUS='USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED'
QUALITY='GENERATED_LDM100K_32FAMILY_FIVESEED_FROZEN_TEACHER15000_STRICT'
p=argparse.ArgumentParser();p.add_argument('--request',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.stage='full'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_bytes())
def own(path,root):
 p=pathlib.Path(path);assert p.is_absolute() and p.resolve()==p and (root==p or root in p.parents)
 assert p.stat().st_uid==1013 and not p.is_symlink();return p
assert os.getuid()==1013 and a.headless and a.device=='cuda:1' and set(os.sched_getaffinity(0))=={0,1,2,3}
request=own(a.request,ROOT);s=read(request);assert sha(__file__)==s['executor_sha256']
for name,digest in s['bound_sources'].items():assert sha(own(ROOT/name,ROOT))==digest,name
fkpath=own(s['fk_receipt'],HOME);assert sha(fkpath)==s['fk_receipt_sha256']
fk=read(fkpath);assert fk['status']=='USER_LDM100K_GENERATED_ISAAC_FK_FINISHED' and fk['output_writers_closed']
mp=own(fk['private_manifest']['path'],HOME);assert sha(mp)==fk['private_manifest']['sha256']
original=[json.loads(x) for x in mp.read_text().splitlines()]
assert len(original)==1880 and all(x['status']=='REAL_GENERATED_ISAAC_FK_EXPORTED' for x in original)
rows=[]
for x in original:
 assert x['source_kind']=='REAL_LDM100000_GENERATED_SCORED_FUTURE' and not x['physical_filter_passed']
 path=own(x['file'],HOME);assert sha(path)==x['sha256'] and path.stat().st_size==x['bytes']
 key='ldm100k-generated|'+x['segment_id']+'|'+str(x['seed'])+'|'+x['mode']
 rows.append(dict(x,canonical_source=key,split='val',mirrored=False))
assert len({x['canonical_source'] for x in rows})==1880
eligible=sorted(range(len(rows)),key=lambda i:(rows[i]['frames'],i))
oldspec=read(ROOT/'configs/teacher15000-direct-strict-spec.json')
teacher=load_teacher(ROOT,oldspec);ckmeta=authorize_teacher(oldspec,teacher);training=teacher['training']
checkpoint=pathlib.Path(ckmeta['path']);checkpoint_sha=sha(checkpoint)
assert checkpoint_sha==ckmeta['sha256']==s['policy_checkpoint_sha256'] and checkpoint.stat().st_size==ckmeta['bytes']
spec=dict(oldspec,seed=20261029,full_parallel_envs=16,full_max_control_calls=30000,
          maximum_private_trace_bytes=195580*36*4)
warm=read(ROOT/'configs/tracker-env-spec.json');assert sha(warm['robot_urdf'])==warm['robot_urdf_sha256']
plan=read(ROOT/'configs/filter-training-plan.json')
out=own(s['output_root'],HOME)/s['run'];out.mkdir(mode=0o700,exist_ok=False)
assert shutil.disk_usage(out).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']
receipt=dict(status='STARTING_USER_LDM100K_GENERATED_PHYSICS',run=s['run'],mode='generated_ldm100k_strict',
 stage='full',quality_label=QUALITY,source_sha256=sha(__file__),request_sha256=sha(request),
 input_fk_receipt_sha256=sha(fkpath),input_manifest_sha256=sha(mp),declared_trials=1880,
 reference_sources=1880,replayed_generated_trials=1880,source_selection='all1880 actual generated seed/mode futures; duration order independent of outcomes',
 policy_checkpoint_sha256=checkpoint_sha,policy_checkpoint_completed_iterations=15000,
 policy_checkpoint_optimizer_updates=300000,policy_original_reference_fingerprint=oldspec['expected_reference_fingerprint'],
 optimizer_updates=0,physical_filter_executed=False,dataset_accepted=False,whole_library_filtered=False,
 full_generator_training_started=False,output_writers_closed=False,actual_execution_video_created=False,
 cpu_affinity=sorted(os.sched_getaffinity(0)),final_endpoint_changes_strict_decision=False)
started=time.monotonic();app=None;gym_env=None;traces=[]
def write(path,value):
 temp=path.with_name(path.name+'.partial')
 with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 temp.replace(path)
def save():
 receipt['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',receipt)
try:
 save();app=AppLauncher(a).app
 import torch,gymnasium as gym
 from isaaclab.managers import TerminationTermCfg
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from rsl_rl.runners import OnPolicyRunner
 from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
 from paper_filter_config import make
 from filter_training_runtime import fingerprint
 from strict_tracking_coverage import StrictCoverage
 from apply_tracking_filter_v3 import contact_signature
 from validated_contact_profile import apply as apply_contact_profile
 from direct_strict_trace import ExecutionTrace
 from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
 torch.set_num_threads(4);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
 torch.cuda.set_per_process_memory_fraction(spec['torch_memory_fraction'],torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>20*1024**3
 fp=fingerprint(rows);assert training['reference_fingerprint']==oldspec['expected_reference_fingerprint'];receipt['generated_reference_fingerprint']=fp
 receipt['reference_fingerprint']=fp
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);info=saved['infos']
 assert info['reference_fingerprint']==oldspec['expected_reference_fingerprint'] and info['diagnostic_only'] is False
 assert info['completed_learning_iterations']==15000 and info['actual_optimizer_updates']==300000
 assert all(len(info['recording'][k])==8278 for k in ['raw_failed','raw_success','failed_motion_count','success_motion_count'])
 del saved
 env_count=spec[a.stage+'_parallel_envs'];assert env_count==(8 if a.stage=='smoke' else 16)
 planned_calls=sum(max(rows[i]['frames'] for i in eligible[k:k+env_count])-1 for k in range(0,len(eligible),env_count))
 assert planned_calls<=spec[a.stage+'_max_control_calls']
 assert sum(rows[i]['frames'] for i in eligible)*36*4<=spec['maximum_private_trace_bytes']
 receipt.update(planned_maximum_control_calls=planned_calls,strict_rollouts_per_motion=1,direct_replay_bypasses_historical_ema_selection=True)
 class AssignedReplay(MotionCommand):
  def __init__(self,cfg,env):
   super().__init__(cfg,env);self.assigned_indices=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  def _uniform_sampling(self,env_ids):return self.assigned_indices[env_ids]
 def motion_complete(env,command_name='motion'):
  m=env.command_manager.get_term(command_name);return m.time_steps>=(m.motion_length-2).clamp_min(0)
 cfg,agent=make(warm,[r['file'] for r in rows],out/'usd',phase='strict',num_envs=env_count)
 receipt['contact_profile']=apply_contact_profile(cfg,out/'usd',ROOT,plan['contact_profile'])
 receipt['contact_profile_sha256']=contact_signature(receipt['contact_profile'])
 assert receipt['contact_profile_sha256']==contact_signature(training['contact_profile'])
 assert cfg.terminations.anchor_pos.params['threshold']==spec['strict_pelvis_height_threshold_m']==.25
 assert cfg.terminations.ee_body_pos.params['threshold']==spec['strict_end_effector_height_threshold_m']==.5
 assert cfg.terminations.anchor_ori is not None and cfg.decimation==4
 cfg.commands.motion.class_type=AssignedReplay;cfg.commands.motion.enable_adaptive_sampling=False
 cfg.commands.motion.start_from_zero_step=True;cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={}
 cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.random_static_prob=-1.
 cfg.episode_length_s=max(rows[i]['frames'] for i in eligible)/50+.04
 cfg.terminations.time_out=None;cfg.terminations.motion_complete=TerminationTermCfg(func=motion_complete,time_out=True)
 gym_env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg)
 wrapped=RslRlVecEnvWrapper(gym_env,clip_actions=None);raw=gym_env.unwrapped;motion=raw.command_manager.get_term('motion')
 assert wrapped.get_observations()[0].shape==(env_count,557)
 assert motion.motion.file_lengths.tolist()==[r['frames'] for r in rows]
 assert len(motion.cfg.body_names)==14 and len(motion.robot.joint_names)==29
 runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=a.device)
 runner.load(str(checkpoint),load_optimizer=False);policy=runner.get_inference_policy(device=a.device)
 def prohibit_optimizer_step(*args,**kwargs):raise AssertionError('Frozen strict replay cannot optimize')
 runner.alg.optimizer.step=prohibit_optimizer_step
 def align_reset_reference():
  before=motion.time_steps.clone();n=len(motion.cfg.body_names)
  ap=motion.anchor_pos_w[:,None,:].repeat(1,n,1);aq=motion.anchor_quat_w[:,None,:].repeat(1,n,1)
  rp=motion.robot_anchor_pos_w[:,None,:].repeat(1,n,1).clone();rq=motion.robot_anchor_quat_w[:,None,:].repeat(1,n,1)
  rp[...,2]=ap[...,2];dq=yaw_quat(quat_mul(rq,quat_inv(aq)))
  motion.body_pos_relative_w=rp+quat_apply(dq,motion.body_pos_w-ap);motion.body_quat_relative_w=quat_mul(dq,motion.body_quat_w)
  assert torch.equal(before,motion.time_steps)
 def snapshot(actions):
  values=dict(reference_frame=motion.time_steps[:,None],root_pos=motion.robot.data.root_pos_w-raw.scene.env_origins,
              root_quat_wxyz=motion.robot.data.root_quat_w,joint_pos=motion.robot_joint_pos,
              reference_joint_pos=motion.joint_pos,actions=actions,
              body_pos=motion.robot_body_pos_w-raw.scene.env_origins[:,None,:],
              reference_body_pos=motion.body_pos_relative_w-raw.scene.env_origins[:,None,:])
  return {k:v.detach().cpu().numpy().copy() for k,v in values.items()}
 captured={};capture={'active':False};original_compute=raw.termination_manager.compute
 def capture_before_autoreset(*args,**kwargs):
  result=original_compute(*args,**kwargs)
  if capture['active']:
   captured.clear();captured['terms']={name:raw.termination_manager.get_term(name).clone() for name in raw.termination_manager.active_terms}
   captured['poses']=snapshot(capture['actions']);capture['samples']+=1
  return result
 raw.termination_manager.compute=capture_before_autoreset
 budget={'used':0,'maximum':spec['maximum_private_trace_bytes']};journal_path=out/'strict_trials.private.jsonl';journal_path.touch(exist_ok=False)
 strict_rows={i:{'motion_id':hashlib.sha256(rows[i]['canonical_source'].encode()).hexdigest(),'trials':[]} for i in eligible}
 trace_index=[];control_calls=0;completed_trials=0;failed_trials=0;capture['samples']=0
 termination_counts={name:0 for name in raw.termination_manager.active_terms}
 receipt.update(status='RUNNING_TEACHER15000_STRICT',num_envs=env_count,quick_pilot_executed=a.stage=='smoke',physical_tracking_executed=True)
 save()
 with torch.inference_mode():
  for start in range(0,len(eligible),env_count):
   batch=eligible[start:start+env_count];padded=batch+[batch[-1]]*(env_count-len(batch));capture['active']=False
   motion.assigned_indices.copy_(torch.tensor(padded,device=a.device))
   gym_env.reset();align_reset_reference();observations=raw.observation_manager.compute()
   initial=snapshot(torch.zeros((env_count,29),device=a.device));batch_traces=[]
   # Fix the world alignment at reset, independently of later command alignment.
   ref_anchor=motion.anchor_pos_w.clone();robot_anchor=motion.robot_anchor_pos_w.clone();robot_anchor[:,2]=ref_anchor[:,2]
   rotation=yaw_quat(quat_mul(motion.robot_anchor_quat_w,quat_inv(motion.anchor_quat_w)))
   endpoint_targets=[]
   import numpy as np
   for slot,i in enumerate(batch):
    indices=[rows[i]['robot_body_names'].index(n) for n in motion.cfg.body_names]
    with np.load(rows[i]['file'],allow_pickle=False) as data:last=data['body_pos_w'][-1,indices].copy()
    target=torch.as_tensor(last,device=a.device)+raw.scene.env_origins[slot]
    transformed=robot_anchor[slot]+quat_apply(rotation[slot].expand(len(indices),4),target-ref_anchor[slot])
    endpoint_targets.append((transformed-raw.scene.env_origins[slot]).detach().cpu().numpy())
   for slot,i in enumerate(batch):
    trace=ExecutionTrace(out/'execution_private'/('trial_'+str(start+slot)),strict_rows[i]['motion_id'],rows[i]['frames'],motion.robot.joint_names,budget=budget)
    traces.append(trace);batch_traces.append(trace);trace.append({k:v[slot] for k,v in initial.items()})
   coverage=[StrictCoverage(rows[i]['frames'],0) for i in batch];pending=set(range(len(batch)))
   for _ in range(max(rows[i]['frames'] for i in batch)-1):
    assert control_calls<spec[a.stage+'_max_control_calls'],'Strict call budget exceeded; preserve incomplete evidence'
    actions=policy(observations['policy']);assert torch.isfinite(actions).all()
    capture['actions']=actions;capture['active']=True;before=capture['samples']
    observations,rewards,terminated,truncated,extras=gym_env.step(actions);capture['active']=False
    assert capture['samples']==before+1,'Exactly one pre-autoreset state per control call required'
    control_calls+=1
    # Record terminal physics poses before checking later observations for failure.
    for slot in sorted(pending):batch_traces[slot].append({k:v[slot] for k,v in captured['poses'].items()})
    assert torch.isfinite(observations['policy']).all() and torch.isfinite(rewards).all()
    term=terminated.tolist();end=truncated.tolist()
    for slot in sorted(list(pending)):
     c=coverage[slot];c.observe(terminated=bool(term[slot]),motion_end=bool(end[slot]))
     if c.end_reason!='running':
      for name,flags in captured['terms'].items():termination_counts[name]+=int(flags[slot])
      record=c.record();strict_rows[batch[slot]]['trials'].append(record)
      with journal_path.open('a') as journal:journal.write(json.dumps({'motion_id':strict_rows[batch[slot]]['motion_id'],'trial':record})+'\n');journal.flush()
      trace_record=batch_traces[slot].finish(record)
      distances=np.linalg.norm(captured['poses']['body_pos'][slot]-endpoint_targets[slot],axis=-1)
      assert np.isfinite(distances).all()
      trace_record['final_reference_endpoint_position']=dict(mean_14_body_m=float(distances.mean()),max_14_body_m=float(distances.max()),
       compared_reference_frame=rows[batch[slot]]['frames']-1,actual_control_steps=record['control_steps'],
       full_T_minus_1_coverage=record['complete'],coordinate_alignment='fixed initial pelvis yaw/xy; reference initial z',
       changes_strict_pass_rule=False)
      trace_record.update(segment_id=rows[batch[slot]]['segment_id'],seed=rows[batch[slot]]['seed'],mode=rows[batch[slot]]['mode'])
      trace_index.append(trace_record);pending.remove(slot)
      completed_trials+=int(record['complete']);failed_trials+=int(record['failed'])
    receipt.update(finished_trials=completed_trials+failed_trials,complete_trials=completed_trials,failed_trials=failed_trials,
                   actual_vectorized_control_calls=control_calls,actual_scene_physics_steps=control_calls*4)
    if a.stage=='smoke' or control_calls%50==0 or not pending:save()
    if not pending:break
   assert not pending,'Incomplete motion is not a completed pilot trial'
 strict=dict(status='DIRECT_STRICT_TRIALS_FINISHED',diagnostic_only=a.stage=='smoke',quality_label=QUALITY,
             policy_checkpoint_sha256=checkpoint_sha,reference_fingerprint=fp,contact_profile_sha256=receipt['contact_profile_sha256'],
             motions=list(strict_rows.values()))
 write(out/'strict.private.json',strict);write(out/'executed_traces.private.json',trace_index)
 receipt.update(status=STATUS,output_writers_closed=True,physical_filter_executed=a.stage=='full',finished_trials=completed_trials+failed_trials,
                complete_trials=completed_trials,failed_trials=failed_trials,
                private_trial_journal_sha256=sha(journal_path),private_strict_log_sha256=sha(out/'strict.private.json'),
                private_executed_trace_index_sha256=sha(out/'executed_traces.private.json'),private_trace_bytes=budget['used'],
                actual_vectorized_control_calls=control_calls,actual_scene_physics_steps=control_calls*4,
                terminal_execution_state_captured_before_autoreset=True,termination_term_event_counts=termination_counts,
                reference_timing='initial0; consumed current targets0..control_steps-1; command advances after termination/reset',
                final_reference_endpoint_comparison_performed=True,
                completion_branch_exercised=completed_trials>0,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                limitation='Frozen15000 policy; one strict T-1 replay per actual generated trial. Independent final-reference position distance is descriptive and does not alter strict pass rules. Not semantic, full-val, WBC/online or paper-complete acceptance.')
 save();print(STATUS,flush=True)
except BaseException as error:
 for trace in traces:trace.close_incomplete()
 receipt.update(status='FAILED_TEACHER15000_STRICT',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=gym_env.unwrapped.sim if gym_env is not None else None,successful_statuses=(STATUS,))
