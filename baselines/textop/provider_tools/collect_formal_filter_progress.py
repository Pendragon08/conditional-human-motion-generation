"""Read bounded aggregate progress; never stop or restart the formal trainer."""
import json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];run='filter_policy_paper69_solid_floor_20261026_v1'
script=r"""
import pathlib,json,os,shutil,subprocess,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN
p=root/'tracker_validation'/run/'receipt.json'
r=json.loads(p.read_text()) if p.exists() else {'status':'NO_TRAINER_RECEIPT_YET','optimizer_updates':0,'completed_learning_iterations':0}
keep=['status','optimizer_updates','completed_learning_iterations','formal_filter_training_started','full_generator_training_started','physical_filter_executed','dataset_accepted','source_sha256','runtime_sha256','plan_sha256','reference_sources','num_envs','reference_fingerprint','target_learning_iterations','latest_losses','raw_completed_attempts','peak_torch_allocated_bytes','storage_preflight','checkpoint_storage_check','latest_checkpoint','contact_profile','paper_settings','error','traceback','output_writers_closed','elapsed_seconds']
r={k:r[k] for k in keep if k in r};r.update(sample_unix=time.time(),run=run,data_free_bytes=shutil.disk_usage(root).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)
launch=root/'receipts'/('filter_policy_'+run+'_launch.json');exitpath=root/'receipts'/('filter_policy_'+run+'_exit.json')
r['launch']=json.loads(launch.read_text()) if launch.exists() else None;r['exit']=json.loads(exitpath.read_text()) if exitpath.exists() else None
r['gpu_resources']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
r['host_memory']=[x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')]
r['personal_task_processes']=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=1013:continue
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(str(root/'scripts'/n) in args for n in ['train_filter_policy.py','supervise_filter_policy.py','train_official.py','launch_training.py']):
   r['personal_task_processes'].append({'pid':int(p.name),'argv':args,'state':[x for x in (p/'status').read_text().splitlines() if x.startswith('State:')][0],'affinity':sorted(os.sched_getaffinity(int(p.name)))})
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(r))
""".replace('RUN',repr(run))
r=json.loads(remote(script,timeout=60));out=B/'results/full_contact_20261026/formal_launch';out.mkdir(exist_ok=True)
(out/'progress.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
d=pathlib.Path('D:/具身智能/TextOp复现_20261024');(d/'正式特权策略训练进度_20261026.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps({k:r.get(k) for k in ['status','completed_learning_iterations','optimizer_updates','latest_losses','data_free_bytes','exit']}))
