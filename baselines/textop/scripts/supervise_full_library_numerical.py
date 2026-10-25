"""Bound one complete-library capacity run; never replace existing workers."""
import argparse,fcntl,json,os,pathlib,re,shutil,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
assert cap['launch_enabled'] is True,'No job started: full-library prerequisites pending'
assert cap['learning_iterations']==1 and cap['supervisor_timeout_seconds']==1800
out=ROOT/'tracker_validation'/a.run;proofpath=ROOT/'receipts'/('full_library_numerical_'+a.run+'_exit.json')
assert not out.exists() and not proofpath.exists(),'Immutable run already exists'
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
# No duplicate SDK job. Paused old generator processes are deliberately retained.
blocked={str(ROOT/'scripts'/x) for x in ['full_library_capacity_probe.py','full_library_numerical_probe.py','supervise_full_library_capacity.py','supervise_full_library_numerical.py','train_filter_policy.py','export_tracker_references.py']}
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not blocked.intersection(argv), 'Existing personal SDK worker; do not duplicate'
memory={x.split(':')[0]:int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
assert memory['MemAvailable']>=cap['minimum_available_host_ram_gib']*1024**3
assert ROOT.stat().st_uid==1013 and shutil.disk_usage(ROOT).free>=cap['minimum_free_data_disk_gib']*1024**3
devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free_gpu={int(x.split(',')[0]):int(x.split(',')[1])*1024**2 for x in devices.strip().splitlines()}
assert free_gpu[1]>=cap['minimum_free_gpu_gib']*1024**3
samples=[];finished=threading.Event();start=time.monotonic()
def sample():
 while not finished.is_set():
  try:
   result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
   samples.append({'elapsed_seconds':time.monotonic()-start,'exit_code':result.returncode,
                   'device_memory_mib_and_utilization_including_other_processes':result.stdout.strip()})
  except subprocess.TimeoutExpired:samples.append({'elapsed_seconds':time.monotonic()-start,'sampling_timeout':True})
  finished.wait(3)
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/full_library_numerical_probe.py'),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--capacity-spec',str(ROOT/'configs/full-library-capacity-spec.json'),
     '--diagnostic-spec',str(ROOT/'configs/full-library-numerical-spec.json'),'--output',str(out),'--device','cuda:1','--headless']
timed_out=False;watch=threading.Thread(target=sample,daemon=True);watch.start()
with (ROOT/'logs'/('full_library_numerical_'+a.run+'.log')).open('xb') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                        preexec_fn=lambda:os.sched_setaffinity(0,{0,1,2,3}))
 try:code=child.wait(timeout=cap['supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
finished.set();watch.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
passed=code==0 and not timed_out and receipt.get('status')=='TEXTOP_FULL_LIBRARY_NUMERICAL_TRACE_COMPLETED' and receipt.get('output_writers_closed') is True
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,
       'numerical_status':receipt.get('status'),'command':cmd,'device_memory_samples':samples,
       'numerical_trace_completed':passed,'full_corpus_stability_accepted':False,'formal_filter_training_started':False,'full_generator_training_started':False}
proofpath.write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
sys.exit(0 if passed else 1)
