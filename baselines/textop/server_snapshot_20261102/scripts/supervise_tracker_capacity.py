"""Bound only the8192 diagnostic; keep process exit and device memory evidence."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
cap=json.loads((ROOT/'configs/tracker-capacity-spec.json').read_text());out=ROOT/'tracker_validation'/a.run
assert not out.exists(),'Immutable run already exists'
samples=[];finished=threading.Event()
def sample():
 while not finished.is_set():
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  samples.append({'elapsed_seconds':time.monotonic()-start,'device_memory_mib_including_other_processes':result.stdout.strip()})
  finished.wait(3)
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/tracker_capacity_probe.py'),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--capacity-spec',str(ROOT/'configs/tracker-capacity-spec.json'),
     '--output',str(out),'--device','cuda:1','--headless']
start=time.monotonic();timed_out=False
watch=threading.Thread(target=sample,daemon=True);watch.start()
with (ROOT/'logs'/('tracker_capacity_'+a.run+'.log')).open('xb') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
     preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
 try:code=child.wait(timeout=cap['supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
finished.set();watch.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,
       'numerical_status':receipt.get('status'),'command':cmd,'device_memory_samples':samples,
       'formal_filter_training_started':False,'full_generator_training_started':False,'full_corpus_capacity_passed':False}
(ROOT/'receipts'/('tracker_capacity_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
if code==0 and not timed_out and receipt.get('status')=='TEXTOP_8192_CAPACITY_PROBE_PASSED':print('TEXTOP_8192_CAPACITY_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
