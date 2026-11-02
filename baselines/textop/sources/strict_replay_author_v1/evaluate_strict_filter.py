"""Whole-motion, original-reference strict replay with actual SDK dynamics.

Diagnostic runs cannot output an acceptance artifact. Formal runs require an
actual completed40k teacher and bind both logs to its checkpoint/reference
fingerprint. Per-motion logs/labels stay private. No optimizer step is taken.
"""
import argparse,hashlib,json,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--replay-spec',required=True)
p.add_argument('--training-plan',required=True);p.add_argument('--mode',choices=['witness','formal'],required=True)
p.add_argument('--checkpoint');p.add_argument('--training-receipt');p.add_argument('--training-exit-receipt');p.add_argument('--output',required=True)
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
 diagnostic=a.mode=='witness'
 assert a.headless and os.getuid()==1013 and a.device==cfgspec['device']=='cuda:1'
 assert len(os.sched_getaffinity(0))<=4 and shutil.disk_usage(out).free>2*1024**3
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==cfgspec['warm_environment_canonical_hash']
 assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
 if not diagnostic:
  assert cfgspec['formal_evaluation_enabled'] and a.checkpoint and a.training_receipt and a.training_exit_receipt
 checkpoint=pathlib.Path(cfgspec['witness_checkpoint'] if diagnostic else a.checkpoint)
 checkpoint_sha=sha(checkpoint)
 if diagnostic:assert checkpoint_sha==cfgspec['witness_checkpoint_sha256']
 else:
  training=json.loads(pathlib.Path(a.training_receipt).read_text());exit_proof=json.loads(pathlib.Path(a.training_exit_receipt).read_text())
  assert training['status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED' and training['mode']=='formal'
  assert training['completed_learning_iterations']==40000 and training['optimizer_updates']==800000
  assert training['num_envs']==8192 and training['latest_checkpoint']['sha256']==checkpoint_sha
  assert exit_proof['child_exit_code']==0 and not exit_proof['timed_out'] and exit_proof['numerical_status']==training['status']
 exports=cfgspec['witness_reference_exports'] if diagnostic else plan['input_reference_exports'];rows=[]
 for name in exports:
  folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
  meta=json.loads((folder/'receipt.json').read_text());assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
  assert sha(folder/'manifest.private.jsonl')==meta['private_manifest_sha256']
  rows.extend(json.loads(x) for x in (folder/'manifest.private.jsonl').read_text().splitlines() if not json.loads(x)['mirrored'])
 assert rows and len({x['canonical_source'] for x in rows})==len(rows)
 assert len(rows)==(cfgspec['witness_expected_sources'] if diagnostic else plan['available_public_source_target'])
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
 torch.set_num_threads(4);torch.manual_seed(cfgspec['seed']);torch.cuda.manual_seed_all(cfgspec['seed'])
 torch.cuda.set_per_process_memory_fraction(cfgspec['torch_memory_fraction'],torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>20*1024**3
 fp=fingerprint(rows);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);info=saved['infos']
 assert info['reference_fingerprint']==fp and info['diagnostic_only']==diagnostic
 completed=info['completed_learning_iterations'];assert completed==(2 if diagnostic else 40000)
 recording=info['recording'];relaxed_rows=[];eligible=[]
 assert all(len(recording[k])==len(rows) for k in ['raw_failed','raw_success','failed_motion_count','success_motion_count'])
 for index,row in enumerate(rows):
  ef=float(recording['failed_motion_count'][index]);es=float(recording['success_motion_count'][index]);assert ef>=0 and es>=0 and ef+es>0
  failed=int(recording['raw_failed'][index]);success=int(recording['raw_success'][index]);assert failed>=0 and success>=0
  motion_id=hashlib.sha256(row['canonical_source'].encode()).hexdigest()
  entry={'motion_id':motion_id,'raw_failed':failed,'raw_success':success,'ema_failed':ef,'ema_success':es,
         'relaxed_failure_probability':ef/(ef+es),'reference_frames':row['frames']}
  relaxed_rows.append(entry)
  if diagnostic or (failed+success>0 and entry['relaxed_failure_probability']<=.05):eligible.append(index)
 if not diagnostic:
  assert all(x['raw_failed']+x['raw_success']>0 for x in relaxed_rows),'Full policy left unattempted references'
  assert training['reference_fingerprint']==fp and training['reference_sources']==len(rows)
 del saved
 relaxed={'status':'DIAGNOSTIC_CHECKPOINT_STATISTICS_ONLY' if diagnostic else 'RELAXED_TRAINED_POLICY_STATISTICS_EXPORTED',
          'policy_checkpoint_sha256':checkpoint_sha,'reference_fingerprint':fp,'motions':relaxed_rows}
 relaxed_path=out/'relaxed.private.json';write(relaxed_path,relaxed)
 env_count=min(cfgspec['maximum_parallel_envs'],max(1,len(eligible)))
 class AssignedReplay(MotionCommand):
  def __init__(self,cfg,env):
   super().__init__(cfg,env);self.assigned_indices=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  def _uniform_sampling(self,env_ids):return self.assigned_indices[env_ids]
 def motion_complete(env,command_name='motion'):
  motion=env.command_manager.get_term(command_name)
  return motion.time_steps>=(motion.motion_length-2).clamp_min(0)
 cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='strict',num_envs=env_count)
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
 strict_rows={index:{'motion_id':relaxed_rows[index]['motion_id'],'trials':[]} for index in eligible}
 control_calls=0;physical_steps=0;completed_trials=0;failed_trials=0;max_contiguous=0
 receipt.update(status='RUNNING_DIAGNOSTIC_STRICT_REPLAY' if diagnostic else 'RUNNING_FORMAL_STRICT_REPLAY',
                num_envs=env_count,eligible_original_sources=len(eligible),policy_checkpoint_completed_iterations=completed)
 save()
 with torch.inference_mode():
  for repeat in range(cfgspec['strict_rollouts_per_motion']):
   for start in range(0,len(eligible),env_count):
    batch=eligible[start:start+env_count];padded=batch+[batch[-1]]*(env_count-len(batch))
    motion.assigned_indices.copy_(torch.tensor(padded,device=a.device))
    observations,_=gym_env.reset()
    coverage=[StrictCoverage(rows[index]['frames'],repeat) for index in batch];pending=set(range(len(batch)))
    for _ in range(max(rows[index]['frames'] for index in batch)-1):
     actions=policy(observations['policy']);assert torch.isfinite(actions).all()
     observations,rewards,terminated,truncated,extras=gym_env.step(actions)
     assert torch.isfinite(observations['policy']).all() and torch.isfinite(rewards).all()
     control_calls+=1;physical_steps+=cfg.decimation
     term=terminated.tolist();end=truncated.tolist()
     for slot in list(pending):
      c=coverage[slot];c.observe(terminated=bool(term[slot]),motion_end=bool(end[slot]))
      if c.end_reason!='running':
       record=c.record();strict_rows[batch[slot]]['trials'].append(record);pending.remove(slot)
       completed_trials+=int(record['complete']);failed_trials+=int(record['failed']);max_contiguous=max(max_contiguous,c.steps)
     if not pending:break
    assert not pending,'Incomplete whole-motion rollout; no acceptance artifact emitted'
    receipt.update(finished_trials=completed_trials+failed_trials,failed_trials=failed_trials,complete_trials=completed_trials,
                   actual_vectorized_control_calls=control_calls);save()
 strict={'status':'STRICT_REPLAY_DIAGNOSTIC_FINISHED' if diagnostic else 'STRICT_ENTIRE_MOTION_ROLLOUTS_FINISHED',
         'diagnostic_only':diagnostic,'policy_checkpoint_sha256':checkpoint_sha,'reference_fingerprint':fp,
         'standard_terminations_restored':True,'pelvis_height_threshold_m':.25,'end_effector_height_threshold_m':.5,
         'reset_distribution':cfgspec['reset_perturbations'],'motions':list(strict_rows.values())}
 strict_path=out/'strict.private.json';write(strict_path,strict)
 if not diagnostic:
  evidence={'protocol':{'schema':'textop.dynamic_filter.v2','privileged_actor_and_critic':True,'domain_randomization':False,
      'training_iterations':40000,'num_envs':8192,'relaxed_removed_terminations':['anchor_pos','ee_body_pos'],
      'strict_standard_terminations_restored':True,'failure_average_definition':'pinned_ema_failed_over_failed_plus_success_alpha0.001',
      'strict_rollouts_per_motion':cfgspec['strict_rollouts_per_motion'],'strict_reset_start_frame':0,'control_hz':50,
      'strict_pelvis_height_threshold_m':.25,'strict_end_effector_height_threshold_m':.5}}
  for key,path in [('policy_checkpoint',checkpoint),('training_receipt',pathlib.Path(a.training_receipt)),
    ('training_exit_receipt',pathlib.Path(a.training_exit_receipt)),('relaxed_rollout_log',relaxed_path),('strict_rollout_log',strict_path)]:
   evidence[key]={'path':str(path),'sha256':sha(path)}
  from apply_tracking_filter import apply
  decisions=apply(evidence);write(out/'evidence.private.json',evidence);write(out/'decisions.private.json',decisions)
  receipt['retained_original_sources']=sum(x['decision']=='retain' for x in decisions)
 receipt.update(status='TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED' if diagnostic else 'TEXTOP_TWO_ROUND_FILTER_EXECUTION_FINISHED',
    finished_trials=completed_trials+failed_trials,failed_trials=failed_trials,complete_trials=completed_trials,
    actual_vectorized_control_calls=control_calls,actual_scene_physics_steps=physical_steps,max_contiguous_control_steps=max_contiguous,
    diagnostic_dynamic_rollouts_executed=diagnostic,physical_filter_executed=not diagnostic,output_writers_closed=True,
    private_relaxed_log_sha256=sha(relaxed_path),private_strict_log_sha256=sha(strict_path),
    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
    completion_branch_exercised=completed_trials>0,limitation=cfgspec['limitation'] if diagnostic else plan['corpus_scope'])
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=gym_env.unwrapped.sim if gym_env is not None else None,
    successful_statuses=('TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED','TEXTOP_TWO_ROUND_FILTER_EXECUTION_FINISHED'))
