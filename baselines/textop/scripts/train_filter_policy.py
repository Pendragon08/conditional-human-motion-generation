"""Paper privileged filter-policy trainer, gated before any formal launch.

Witness mode validates the extra statistics and checkpoint runtime only. No
accepted motions or generator launch can result from witness mode.
"""
import argparse,hashlib,json,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--plan',required=True)
p.add_argument('--mode',choices=['witness','formal'],required=True);p.add_argument('--output',required=True);p.add_argument('--resume')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'completed_learning_iterations':0,'formal_filter_training_started':False,
         'full_generator_training_started':False,'physical_filter_executed':False,'dataset_accepted':False,'output_writers_closed':False}
app=None;env=None;runners=[];started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-started
 partial=out/'receipt.json.partial';partial.write_text(json.dumps(receipt,indent=2));partial.replace(out/'receipt.json')
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());plan=json.loads(pathlib.Path(a.plan).read_text())
 assert a.headless and os.getuid()==1013 and a.device==plan['device']=='cuda:1' and len(os.sched_getaffinity(0))<=4
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==plan['warm_environment_canonical_hash']
 assert shutil.disk_usage(out).free>plan['minimum_disk_free_gib']*1024**3
 diagnostic=a.mode=='witness';exports=['export_author_v1'] if diagnostic else plan['input_reference_exports']
 if not diagnostic:assert plan['formal_launch_enabled'] is True and not plan['formal_launch_blockers'],'Formal launch prerequisites not fulfilled'
 rows=[]
 for name in exports:
  ref=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
  meta=json.loads((ref/'receipt.json').read_text());assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
  assert sha(ref/'manifest.private.jsonl')==meta['private_manifest_sha256']
  rows += [json.loads(x) for x in (ref/'manifest.private.jsonl').read_text().splitlines() if not json.loads(x)['mirrored']]
 assert rows and len({x['canonical_source'] for x in rows})==len(rows) and all(sha(x['file'])==x['sha256'] for x in rows)
 if not diagnostic:assert len(rows)==plan['available_public_source_target']
 receipt.update(mode=a.mode,source_sha256=sha(__file__),runtime_sha256=sha(pathlib.Path(__file__).with_name('filter_training_runtime.py')),
       plan_sha256=sha(a.plan),reference_sources=len(rows),num_envs=8 if diagnostic else 8192,corpus_scope=plan['corpus_scope'])
 save();app=AppLauncher(a).app
 import torch,gymnasium as gym
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from rsl_rl.runners import OnPolicyRunner
 from paper_filter_config import make,paper_settings
 from filter_training_runtime import RecordedMotionCommand,fingerprint,save_recorded,load_recorded
 torch.set_num_threads(4);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
 torch.cuda.set_per_process_memory_fraction(plan['torch_memory_fraction'],torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>plan['minimum_gpu_free_gib']*1024**3
 cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=receipt['num_envs'])
 cfg.commands.motion.class_type=RecordedMotionCommand
 env=RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg));motion=env.unwrapped.command_manager.get_term('motion')
 assert env.get_observations()[0].shape==(receipt['num_envs'],557)
 fp=fingerprint(rows);target=2 if diagnostic else plan['learning_iterations'];updates=0;completed=0
 receipt.update(status='RUNNING_STATISTICS_WITNESS' if diagnostic else 'RUNNING_FORMAL_FILTER_TRAINING',
       reference_fingerprint=fp,target_learning_iterations=target,paper_settings=paper_settings(),formal_filter_training_started=not diagnostic)
 # Initial reset must never create an independently counted success or failure.
 assert int((motion.raw_failed+motion.raw_success).sum())==0
 def build(log):
  runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/log),device=a.device);runners.append(runner)
  step=runner.alg.optimizer.step
  def counted(*args,**kwargs):
   global updates
   assert all(torch.isfinite(x).all() and (x.grad is None or torch.isfinite(x.grad).all()) for x in runner.alg.policy.parameters())
   value=step(*args,**kwargs);updates+=1;receipt['optimizer_updates']=updates;return value
  runner.alg.optimizer.step=counted
  return runner
 runner=build('training_logs')
 if a.resume:
  info=load_recorded(runner,motion,a.resume,fp);assert info['diagnostic_only']==diagnostic
  updates=info['actual_optimizer_updates'];completed=info['completed_learning_iterations'];assert updates==completed*20
  receipt['resume_exact_state_verified']=True
 save()
 if diagnostic:
  runner.learn(num_learning_iterations=1,init_at_random_ep_len=False);assert updates==20
  ckpt=out/'completed_1.pt';first=save_recorded(runner,motion,ckpt,1,updates,fp,True)
  receipt.update(completed_learning_iterations=1,checkpoints=[first]);save()
  runner.writer.flush();runner.writer.close();runner.writer=None
  restored=build('restored_training_logs');info=load_recorded(restored,motion,ckpt,fp)
  assert info['completed_learning_iterations']==1 and updates==20
  receipt['resume_exact_state_verified']=True
  restored.learn(num_learning_iterations=1,init_at_random_ep_len=False);assert updates==40
  final=save_recorded(restored,motion,out/'completed_2.pt',2,updates,fp,True)
  receipt.update(status='TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED',completed_learning_iterations=2,
                 checkpoints=[first,final],initial_resets_excluded_from_raw_counts=True,
                 raw_completed_attempts=int((motion.raw_failed+motion.raw_success).sum()),
                 limitation='8envs/2iterations. Exact model/Adam/normalizers/EMA/raw-counters restore only; not full policy or accepted motions')
 else:
  # Pinned PPO executes unchanged; log callback supplies exact post-update
  # counters and safe completed-iteration checkpoints, avoiding loop-index ambiguity.
  original_log=runner.log;original_save=runner.save
  def monitored_log(values,*args,**kwargs):
   global completed
   original_log(values,*args,**kwargs);assert updates%20==0;completed=updates//20
   assert completed==values['it']+1
   losses={k:float(v) for k,v in values['loss_dict'].items()};assert all(__import__('math').isfinite(x) for x in losses.values())
   receipt.update(completed_learning_iterations=completed,optimizer_updates=updates,latest_losses=losses,
                  raw_completed_attempts=int((motion.raw_failed+motion.raw_success).sum()),
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
   if completed%100==0:
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'completed':completed,'updates':updates,'losses':losses})+'\n')
    save()
   if completed%plan['checkpoint_interval']==0 or completed==1000 or shutil.disk_usage(out).free<plan['minimum_disk_free_gib']*1024**3:
    runner.save=original_save
    try:receipt['latest_checkpoint']=save_recorded(runner,motion,out/('completed_'+str(completed)+'.pt'),completed,updates,fp,False)
    finally:runner.save=suppressed_save
    save()
   assert shutil.disk_usage(out).free>=plan['minimum_disk_free_gib']*1024**3,'Disk guard; safe checkpoint saved, existing data retained'
  def suppressed_save(*args,**kwargs):pass
  runner.log=monitored_log;runner.save=suppressed_save
  runner.learn(num_learning_iterations=target-completed,init_at_random_ep_len=False)
  assert completed==target and updates==target*20
  receipt.update(status='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED',completed_learning_iterations=completed)
 for runner in runners:
  if runner.writer is not None:runner.writer.flush();runner.writer.close();runner.writer=None
 receipt.update(output_writers_closed=True,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device));save()
 print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 for runner in runners:
  if runner.writer is not None:runner.writer.flush();runner.writer.close();runner.writer=None
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
        successful_statuses=('TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED','TEXTOP_FORMAL_FILTER_TRAINING_FINISHED'))
