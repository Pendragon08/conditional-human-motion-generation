"""Supervise only the explicitly requested private filter-policy child."""
import argparse,fcntl,json,os,pathlib,re,shutil,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--mode',required=True,choices=['witness','formal']);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text())
if a.mode=='formal':assert plan['formal_launch_enabled'] and not plan['formal_launch_blockers']
out=ROOT/'tracker_validation'/a.run;assert not out.exists(),'Immutable run already exists'
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 blocked=['run_tracker.sh','train_filter_policy.py','export_tracker_references.py','full_library_capacity_probe.py',
          'full_library_numerical_probe.py','contact_replay_probe.py','full_contact_probe.py','full_solid_floor_probe.py','solid_floor_ppo_probe.py']
 assert not {str(ROOT/'scripts'/x) for x in blocked}.intersection(argv),'Existing personal SDK worker'
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
capacity=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
assert mem>=capacity['minimum_available_host_ram_gib']*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1])*1024**2 for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()}
assert free[1]>=plan['minimum_gpu_free_gib']*1024**3
assert shutil.disk_usage(ROOT).free>plan['minimum_disk_free_gib']*1024**3

cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/train_filter_policy.py'),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--plan',str(ROOT/'configs/filter-training-plan.json'),
     '--output',str(out),'--mode',a.mode,'--device','cuda:1','--headless']
started=time.monotonic();samples=[];done=threading.Event();timed_out=False
def sample():
 while not done.is_set():
  proc=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  samples.append({'elapsed_seconds':time.monotonic()-started,'device_memory_mib_including_other_processes':proc.stdout.strip()});done.wait(3 if a.mode=='witness' else 60)
thread=threading.Thread(target=sample,daemon=True);thread.start()
with (ROOT/'logs'/('filter_policy_'+a.run+'.log')).open('xb') as target:
 child=subprocess.Popen(cmd,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
     preexec_fn=lambda:os.sched_setaffinity(0,{0,1,2,3}))
 (ROOT/'receipts'/('filter_policy_'+a.run+'_launch.json')).write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'command':cmd,'mode':a.mode},indent=2))
 try:code=child.wait(timeout=600 if a.mode=='witness' else 7*86400)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
done.set();thread.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
expected='TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED' if a.mode=='witness' else 'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED'
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,
       'numerical_status':receipt.get('status'),'command':cmd,'device_memory_samples':samples,
       'formal_filter_training_started':receipt.get('formal_filter_training_started',False),'full_generator_training_started':False,'dataset_accepted':False}
(ROOT/'receipts'/('filter_policy_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
if code==0 and not timed_out and receipt.get('status')==expected:print('TEXTOP_FILTER_POLICY_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
