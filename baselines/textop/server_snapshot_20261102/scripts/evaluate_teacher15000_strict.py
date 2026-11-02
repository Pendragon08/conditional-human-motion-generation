"""User-directed frozen15k strict complete replay, with no training."""
import argparse, hashlib, json, os, pathlib, shutil, time, traceback
from isaaclab.app import AppLauncher
from teacher15000_strict_protocol import gate, load_teacher, authorize_teacher, select_sources, settings_signature, verify_smoke_witnesses, QUALITY, STATUS
ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p = argparse.ArgumentParser();p.add_argument('--pilot-spec',required=True)
p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
spec=json.loads(pathlib.Path(a.pilot_spec).read_text());gate(a.stage,spec)
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt=dict(status='STARTING_TEACHER15000_STRICT',mode='teacher15000_strict',stage=a.stage,quality_label=QUALITY,
             optimizer_updates=0,physical_filter_executed=False,dataset_accepted=False,
             whole_library_filtered=False,full_generator_training_started=False,output_writers_closed=False,
             actual_execution_video_created=False)
started=time.monotonic();app=None;gym_env=None;traces=[]
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8388608),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def save():
 receipt['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',receipt)
try:
 teacher=load_teacher(ROOT,spec);ckmeta=authorize_teacher(spec,teacher);training=teacher['training']
 checkpoint=pathlib.Path(ckmeta['path']);checkpoint_sha=sha(checkpoint)
 assert checkpoint_sha==ckmeta['sha256'] and checkpoint.stat().st_size==ckmeta['bytes']
 assert os.getuid()==1013 and a.headless and a.device==spec['device']=='cuda:1'
 assert set(os.sched_getaffinity(0))=={0,1,2,3} and shutil.disk_usage(out).free>2*1024**3
 for name,key in [('evaluate_teacher15000_strict.py','executor_sha256'),('teacher15000_strict_protocol.py','protocol_sha256'),('direct_strict_trace.py','trace_sha256')]:
  assert sha(ROOT/'scripts'/name)==spec[key]
 if a.stage=='full':verify_smoke_witnesses(ROOT,spec,checkpoint_sha)
 warm=json.loads((ROOT/'configs/tracker-env-spec.json').read_text())
 assert hashlib.sha256(json.dumps(warm,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==spec['warm_environment_canonical_hash']
 assert sha(warm['robot_urdf'])==warm['robot_urdf_sha256']
 plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());rows=[]
 for name in plan['input_reference_exports']:
  folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
  meta=json.loads((folder/'receipt.json').read_text());assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
  assert sha(folder/'manifest.private.jsonl')==meta['private_manifest_sha256']
  rows.extend(r for r in map(json.loads,(folder/'manifest.private.jsonl').read_text().splitlines()) if not r['mirrored'])
 assert len(rows)==8278 and len({r['canonical_source'] for r in rows})==8278
 assert all(sha(r['file'])==r['sha256'] and r['frames']>=2 for r in rows)
 eligible=select_sources(rows,a.stage,spec)
 receipt.update(source_sha256=sha(__file__),evaluation_protocol_sha256=spec['protocol_sha256'],trace_source_sha256=spec['trace_sha256'],
                pilot_spec_sha256=sha(a.pilot_spec),execution_settings_sha256=settings_signature(spec),
                teacher_artifact_sha256=teacher['artifact_sha256'],policy_checkpoint_sha256=checkpoint_sha,
                policy_checkpoint_completed_iterations=15000,policy_checkpoint_optimizer_updates=300000,
                reference_sources=len(rows),replayed_original_sources=len(eligible),
                source_selection='eight shortest whole originals50..500frames' if a.stage=='smoke' else 'All8278 originals once; sorted by (duration, original index); direct strict replay without EMA selection',
                unattempted_original_sources=len(rows)-len(eligible),cpu_affinity=sorted(os.sched_getaffinity(0)))
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
 fp=fingerprint(rows);assert fp==spec['expected_reference_fingerprint']==training['reference_fingerprint']
 receipt['reference_fingerprint']=fp
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);info=saved['infos']
 assert info['reference_fingerprint']==fp and info['diagnostic_only'] is False
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
      trace_index.append(batch_traces[slot].finish(record));pending.remove(slot)
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
                final_reference_endpoint_comparison_performed=False,
                completion_branch_exercised=completed_trials>0,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                limitation='Frozen15000, one deterministic strict complete replay per original. Direct user route bypasses historical EMA selection. Final reference endpoint and semantic/paper quality are not established. Corpus acceptance is separate.')
 save();print(STATUS,flush=True)
except BaseException as error:
 for trace in traces:trace.close_incomplete()
 receipt.update(status='FAILED_TEACHER15000_STRICT',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=gym_env.unwrapped.sim if gym_env is not None else None,successful_statuses=(STATUS,))
