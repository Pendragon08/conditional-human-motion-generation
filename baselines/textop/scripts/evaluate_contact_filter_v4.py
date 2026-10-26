"""Executor revision4: bounded validation and contact-consistent formal v3 export.

Diagnostic runs cannot output an acceptance artifact. Formal runs require an
actual completed40k teacher and bind both logs to its checkpoint/reference
fingerprint. Per-motion logs/labels stay private. No optimizer step is taken.
"""
import argparse,hashlib,json,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--replay-spec',required=True)
p.add_argument('--training-plan',required=True);p.add_argument('--mode',choices=['formal','validation'],required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--training-receipt',required=True);p.add_argument('--training-exit-receipt',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','mode':a.mode,'optimizer_updates':0,'formal_filter_training_started':False,
 'physical_filter_executed':False,'diagnostic_dynamic_rollouts_executed':False,'dataset_accepted':False,
 'full_generator_training_started':False,'output_writers_closed':False}
app=None;gym_env=None;started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8388608),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def save():
 receipt['elapsed_seconds']=time.monotonic()-started;write(out/'receipt.json',receipt)
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());cfgspec=json.loads(pathlib.Path(a.replay_spec).read_text());plan=json.loads(pathlib.Path(a.training_plan).read_text())
 from contact_replay_modes import authorize,gate_mode,select_validation_sources,finish_status,VALIDATION_STATUS
 gate_mode(a.mode,cfgspec)
 training=json.loads(pathlib.Path(a.training_receipt).read_text());exit_proof=json.loads(pathlib.Path(a.training_exit_receipt).read_text())
 diagnostic=authorize(a.mode,cfgspec,training,exit_proof)
 assert a.headless and os.getuid()==1013 and a.device==cfgspec['device']=='cuda:1'
 assert set(os.sched_getaffinity(0))=={0,1,2,3} and shutil.disk_usage(out).free>2*1024**3
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==cfgspec['warm_environment_canonical_hash']
 assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
 checkpoint=pathlib.Path(a.checkpoint);checkpoint_sha=sha(checkpoint)
 assert training['plan_sha256']==sha(a.training_plan)
 assert cfgspec['acceptor_sha256']==sha(pathlib.Path(__file__).with_name('apply_tracking_filter_v3.py'))
 assert cfgspec['mode_runtime_sha256']==sha(pathlib.Path(__file__).with_name('contact_replay_modes.py'))
 assert cfgspec['executor_sha256']==sha(__file__) and cfgspec['acceptance_schema']=='textop.dynamic_filter.v3'
 assert training['latest_checkpoint']['sha256']==checkpoint_sha
 exports=plan['input_reference_exports'];rows=[]
 for name in exports:
  folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
  meta=json.loads((folder/'receipt.json').read_text());assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
  assert sha(folder/'manifest.private.jsonl')==meta['private_manifest_sha256']
  rows.extend(json.loads(x) for x in (folder/'manifest.private.jsonl').read_text().splitlines() if not json.loads(x)['mirrored'])
 assert rows and len({x['canonical_source'] for x in rows})==len(rows)
 assert len(rows)==plan['available_public_source_target']
 assert all(sha(x['file'])==x['sha256'] and x['frames']>=2 for x in rows)
 receipt.update(source_sha256=sha(__file__),replay_spec_sha256=sha(a.replay_spec),reference_sources=len(rows),
   policy_checkpoint_sha256=checkpoint_sha,cpu_affinity=sorted(os.sched_getaffinity(0)),
   deterministic_reset=True,strict_pelvis_height_threshold_m=.25,strict_end_effector_height_threshold_m=.5)
 save();app=AppLauncher(a).app
 import torch,gymnasium as gym
 from isaaclab.managers import TerminationTermCfg
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from rsl_rl.runners import OnPolicyRunner
 from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
 from paper_filter_config import make
 from filter_training_runtime import fingerprint
 from strict_tracking_coverage import StrictCoverage
 from apply_tracking_filter_v3 import contact_signature,require_completed_training
 from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
 torch.set_num_threads(4);torch.manual_seed(cfgspec['seed']);torch.cuda.manual_seed_all(cfgspec['seed'])
 torch.cuda.set_per_process_memory_fraction(cfgspec['torch_memory_fraction'],torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>20*1024**3
 fp=fingerprint(rows);receipt['reference_fingerprint']=fp
 receipt.update(training_receipt_sha256=sha(a.training_receipt),training_plan_sha256=sha(a.training_plan))
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);info=saved['infos']
 assert info['reference_fingerprint']==fp and info['diagnostic_only'] is False
 completed=info['completed_learning_iterations'];assert completed==40000
 recording=info['recording'];relaxed_rows=[];eligible=[]
 assert all(len(recording[k])==len(rows) for k in ['raw_failed','raw_success','failed_motion_count','success_motion_count'])
 for index,row in enumerate(rows):
  ef=float(recording['failed_motion_count'][index]);es=float(recording['success_motion_count'][index]);assert ef>=0 and es>=0 and ef+es>0
  failed=int(recording['raw_failed'][index]);success=int(recording['raw_success'][index]);assert failed>=0 and success>=0
  motion_id=hashlib.sha256(row['canonical_source'].encode()).hexdigest()
  entry={'motion_id':motion_id,'raw_failed':failed,'raw_success':success,'ema_failed':ef,'ema_success':es,
         'relaxed_failure_probability':ef/(ef+es),'reference_frames':row['frames']}
  relaxed_rows.append(entry)
  if failed+success>0 and entry['relaxed_failure_probability']<=.05:eligible.append(index)
 assert all(x['raw_failed']+x['raw_success']>0 for x in relaxed_rows),'Full policy left unattempted references'
 assert training['reference_fingerprint']==fp and training['reference_sources']==len(rows)
 if diagnostic:
  eligible=select_validation_sources(rows,cfgspec)
  receipt.update(validation_source_selection='eight shortest complete original sources within50..500frames; stable global-index tie-break; no failure-rate selection',validation_source_count=len(eligible),validation_max_control_calls=cfgspec['validation_max_control_calls'])
 del saved
 relaxed={'status':'DIAGNOSTIC_COMPLETED_TEACHER_STATISTICS_ONLY' if diagnostic else 'RELAXED_TRAINED_POLICY_STATISTICS_EXPORTED',
          'policy_checkpoint_sha256':checkpoint_sha,'reference_fingerprint':fp,'motions':relaxed_rows}
 relaxed_path=out/'relaxed.private.json';write(relaxed_path,relaxed)
 env_count=min(cfgspec['validation_maximum_parallel_envs'] if diagnostic else cfgspec['maximum_parallel_envs'],max(1,len(eligible)))
 class AssignedReplay(MotionCommand):
  def __init__(self,cfg,env):
   super().__init__(cfg,env);self.assigned_indices=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  def _uniform_sampling(self,env_ids):return self.assigned_indices[env_ids]
 def motion_complete(env,command_name='motion'):
  motion=env.command_manager.get_term(command_name)
  return motion.time_steps>=(motion.motion_length-2).clamp_min(0)
 cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='strict',num_envs=env_count)
 from validated_contact_profile import apply as apply_contact_profile
 receipt['contact_profile']=apply_contact_profile(cfg,out/'usd',pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024'),plan['contact_profile'])
 receipt['contact_profile_sha256']=contact_signature(receipt['contact_profile'])
 assert receipt['contact_profile_sha256']==contact_signature(training['contact_profile'])
 relaxed['contact_profile_sha256']=receipt['contact_profile_sha256'];write(relaxed_path,relaxed)
 assert cfg.terminations.anchor_pos.params['threshold']==.25 and cfg.terminations.ee_body_pos.params['threshold']==.5
 assert cfg.terminations.anchor_ori is not None
 cfg.commands.motion.class_type=AssignedReplay;cfg.commands.motion.enable_adaptive_sampling=False
 cfg.commands.motion.start_from_zero_step=True;cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={}
 cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.random_static_prob=-1.
 # End exactly after the last transition; default10stimeout cannot pass a long motion.
 cfg.episode_length_s=max(x['frames'] for x in rows)/50+.04
 cfg.terminations.time_out=None
 cfg.terminations.motion_complete=TerminationTermCfg(func=motion_complete,time_out=True)
 gym_env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg)
 wrapped=RslRlVecEnvWrapper(gym_env,clip_actions=None);raw=gym_env.unwrapped;motion=raw.command_manager.get_term('motion')
 assert wrapped.get_observations()[0].shape==(env_count,557)
 assert motion.motion.file_lengths.tolist()==[x['frames'] for x in rows]
 runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=a.device)
 runner.load(str(checkpoint),load_optimizer=False);policy=runner.get_inference_policy(device=a.device)
 # This evaluator must never take an optimizer step.
 def prohibit_optimizer_step(*args,**kwargs):raise AssertionError('Optimizer step prohibited in evaluator')
 runner.alg.optimizer.step=prohibit_optimizer_step
 journal_path=out/'strict_trials.private.jsonl'
 journal_path.touch(exist_ok=False)
 strict_rows={index:{'motion_id':relaxed_rows[index]['motion_id'],'trials':[]} for index in eligible}
 # Pinned MotionCommand initializes its aligned-reference buffers to zeros and
 # refreshes them only AFTER termination computation in a normal step. Refresh
 # that exact transform after reset, without advancing time or EMA, so first-
 # step height tests don't compare the robot against stale zero/previous data.
 def align_reset_reference():
  before=motion.time_steps.clone();n=len(motion.cfg.body_names)
  ap=motion.anchor_pos_w[:,None,:].repeat(1,n,1);aq=motion.anchor_quat_w[:,None,:].repeat(1,n,1)
  rp=motion.robot_anchor_pos_w[:,None,:].repeat(1,n,1).clone();rq=motion.robot_anchor_quat_w[:,None,:].repeat(1,n,1)
  rp[...,2]=ap[...,2];dq=yaw_quat(quat_mul(rq,quat_inv(aq)))
  motion.body_pos_relative_w=rp+quat_apply(dq,motion.body_pos_w-ap)
  motion.body_quat_relative_w=quat_mul(dq,motion.body_quat_w)
  assert torch.equal(before,motion.time_steps)
 captured={};original_termination_compute=raw.termination_manager.compute
 def capture_terminations(*args,**kwargs):
  result=original_termination_compute(*args,**kwargs)
  captured.clear();captured.update({name:raw.termination_manager.get_term(name).clone() for name in raw.termination_manager.active_terms})
  return result
 raw.termination_manager.compute=capture_terminations
 termination_counts={name:0 for name in raw.termination_manager.active_terms};initial_max_error=0.
 control_calls=0;physical_steps=0;completed_trials=0;failed_trials=0;max_contiguous=0
 receipt.update(status='RUNNING_DIAGNOSTIC_STRICT_REPLAY' if diagnostic else 'RUNNING_FORMAL_STRICT_REPLAY',
                num_envs=env_count,eligible_original_sources=None if diagnostic else len(eligible),replayed_original_sources=len(eligible),policy_checkpoint_completed_iterations=completed)
 save()
 with torch.inference_mode():
  for repeat in range(cfgspec['strict_rollouts_per_motion']):
   for start in range(0,len(eligible),env_count):
    batch=eligible[start:start+env_count];padded=batch+[batch[-1]]*(env_count-len(batch))
    motion.assigned_indices.copy_(torch.tensor(padded,device=a.device))
    observations,_=gym_env.reset();align_reset_reference();observations=raw.observation_manager.compute()
    initial_max_error=max(initial_max_error,float((motion.body_pos_relative_w-motion.robot_body_pos_w).norm(dim=-1).max()))
    coverage=[StrictCoverage(rows[index]['frames'],repeat) for index in batch];pending=set(range(len(batch)))
    for _ in range(max(rows[index]['frames'] for index in batch)-1):
     if diagnostic:assert control_calls<cfgspec['validation_max_control_calls'],'Validation call budget reached; preserve partial records, no completion claim'
     actions=policy(observations['policy']);assert torch.isfinite(actions).all()
     observations,rewards,terminated,truncated,extras=gym_env.step(actions)
     assert torch.isfinite(observations['policy']).all() and torch.isfinite(rewards).all()
     control_calls+=1;physical_steps+=cfg.decimation
     term=terminated.tolist();end=truncated.tolist()
     for slot in list(pending):
      c=coverage[slot];c.observe(terminated=bool(term[slot]),motion_end=bool(end[slot]))
      if c.end_reason!='running':
       for name,flags in captured.items():termination_counts[name]+=int(flags[slot])
       record=c.record();strict_rows[batch[slot]]['trials'].append(record);pending.remove(slot)
       with journal_path.open('a') as journal:journal.write(json.dumps({'motion_id':relaxed_rows[batch[slot]]['motion_id'],'trial':record})+'\n');journal.flush()
       completed_trials+=int(record['complete']);failed_trials+=int(record['failed']);max_contiguous=max(max_contiguous,c.steps)
     if not pending:break
    assert not pending,'Incomplete whole-motion rollout; no acceptance artifact emitted'
    receipt.update(finished_trials=completed_trials+failed_trials,failed_trials=failed_trials,complete_trials=completed_trials,
                   actual_vectorized_control_calls=control_calls);save()
 strict={'status':'STRICT_REPLAY_DIAGNOSTIC_FINISHED' if diagnostic else 'STRICT_ENTIRE_MOTION_ROLLOUTS_FINISHED',
         'diagnostic_only':diagnostic,'policy_checkpoint_sha256':checkpoint_sha,'reference_fingerprint':fp,
         'standard_terminations_restored':True,'pelvis_height_threshold_m':.25,'end_effector_height_threshold_m':.5,
         'reset_distribution':cfgspec['reset_perturbations'],'motions':list(strict_rows.values())}
 strict['contact_profile_sha256']=receipt['contact_profile_sha256']
 strict_path=out/'strict.private.json';write(strict_path,strict)
 if not diagnostic:
  evidence={'protocol':{'schema':'textop.dynamic_filter.v3','privileged_actor_and_critic':True,'domain_randomization':False,
      'training_iterations':40000,'num_envs':8192,'relaxed_removed_terminations':['anchor_pos','ee_body_pos'],
      'strict_standard_terminations_restored':True,'failure_average_definition':'pinned_ema_failed_over_failed_plus_success_alpha0.001',
      'strict_rollouts_per_motion':cfgspec['strict_rollouts_per_motion'],'strict_reset_start_frame':0,'control_hz':50,
      'strict_pelvis_height_threshold_m':.25,'strict_end_effector_height_threshold_m':.5}}
  for key,path in [('policy_checkpoint',checkpoint),('training_receipt',pathlib.Path(a.training_receipt)),
    ('training_exit_receipt',pathlib.Path(a.training_exit_receipt)),('relaxed_rollout_log',relaxed_path),('strict_rollout_log',strict_path),('training_plan',pathlib.Path(a.training_plan)),
    ('evaluation_source',pathlib.Path(__file__)),('evaluation_spec',pathlib.Path(a.replay_spec)),('strict_trial_journal',journal_path)]:
   evidence[key]={'path':str(path),'sha256':sha(path)}
  # Actual supervisor exit/closed receipt do not exist yet: export inputs only.
  # A separate post-exit finalizer binds these artifacts and runs v3 acceptance.
  write(out/'evidence_inputs.private.json',evidence)
  receipt['post_exit_evidence_finalization_pending']=True
 receipt['private_trial_journal_sha256']=sha(journal_path)
 receipt.update(status=finish_status(a.mode),
    finished_trials=completed_trials+failed_trials,failed_trials=failed_trials,complete_trials=completed_trials,
    actual_vectorized_control_calls=control_calls,actual_scene_physics_steps=physical_steps,max_contiguous_control_steps=max_contiguous,
    diagnostic_dynamic_rollouts_executed=diagnostic,physical_filter_executed=not diagnostic,output_writers_closed=True,
    private_relaxed_log_sha256=sha(relaxed_path),private_strict_log_sha256=sha(strict_path),
    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
    completion_branch_exercised=completed_trials>0,limitation=cfgspec['limitation'] if diagnostic else plan['corpus_scope'])
 receipt.update(reset_reference_alignment_refreshed_without_time_advance=True,initial_max_tracked_body_error_m=initial_max_error,
                termination_term_event_counts=termination_counts,validation_only=diagnostic)
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=gym_env.unwrapped.sim if gym_env is not None else None,
    successful_statuses=(VALIDATION_STATUS,'TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED'))
