"""Bound a diagnostic child, retaining actual exit, log and total GPU samples."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
cfg=json.loads((ROOT/'configs/tracker-initialization-audit.json').read_text());out=ROOT/'tracker_validation'/a.run
assert not out.exists(),'Immutable run already exists'
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/tracker_initialization_audit.py'),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--audit-spec',str(ROOT/'configs/tracker-initialization-audit.json'),
     '--output',str(out),'--device','cuda:1','--headless']
started=time.monotonic();samples=[];done=threading.Event();timed_out=False
def sample():
 while not done.is_set():
  proc=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  samples.append({'elapsed_seconds':time.monotonic()-started,'device_memory_mib_including_other_processes':proc.stdout.strip()});done.wait(3)
thread=threading.Thread(target=sample,daemon=True);thread.start()
with (ROOT/'logs'/('tracker_initialization_'+a.run+'.log')).open('xb') as target:
 child=subprocess.Popen(cmd,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
     preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
 try:code=child.wait(timeout=cfg['supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
done.set();thread.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,
       'numerical_status':receipt.get('status'),'command':cmd,'device_memory_samples':samples,
       'formal_filter_training_started':False,'full_generator_training_started':False,'dataset_accepted':False}
(ROOT/'receipts'/('tracker_initialization_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
if code==0 and not timed_out and receipt.get('status')=='TEXTOP_INITIALIZATION_AUDIT_FINISHED':print('TEXTOP_INITIALIZATION_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
