"""Bound only a named strict replay child; retain actual exit and GPU samples."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--mode',choices=['witness','formal'],required=True)
p.add_argument('--checkpoint');p.add_argument('--training-receipt');p.add_argument('--training-exit-receipt');a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
spec=json.loads((ROOT/'configs/strict-replay-spec.json').read_text());out=ROOT/'tracker_validation'/a.run
assert not out.exists(),'Immutable run already exists'
if a.mode=='formal':assert spec['formal_evaluation_enabled'] and all([a.checkpoint,a.training_receipt,a.training_exit_receipt])
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/evaluate_strict_filter.py'),
 '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--replay-spec',str(ROOT/'configs/strict-replay-spec.json'),
 '--training-plan',str(ROOT/'configs/filter-training-plan.json'),'--mode',a.mode,'--output',str(out),'--device','cuda:1','--headless']
for key,value in [('--checkpoint',a.checkpoint),('--training-receipt',a.training_receipt),('--training-exit-receipt',a.training_exit_receipt)]:
 if value:cmd.extend([key,value])
started=time.monotonic();done=threading.Event();samples=[];timed_out=False
def sample():
 while not done.is_set():
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  samples.append({'elapsed_seconds':time.monotonic()-started,'device_memory_mib_including_other_processes':result.stdout.strip()});done.wait(3 if a.mode=='witness' else 30)
watch=threading.Thread(target=sample,daemon=True);watch.start()
with (ROOT/'logs'/('strict_replay_'+a.run+'.log')).open('xb') as target:
 child=subprocess.Popen(cmd,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
   preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
 (ROOT/'receipts'/('strict_replay_'+a.run+'_launch.json')).write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'command':cmd},indent=2))
 try:code=child.wait(timeout=spec['witness_supervisor_timeout_seconds'] if a.mode=='witness' else spec['formal_supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
done.set();watch.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'mode':a.mode,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,
 'numerical_status':receipt.get('status'),'elapsed_seconds':time.monotonic()-started,'command':cmd,'device_memory_samples':samples,
 'physical_filter_executed':receipt.get('physical_filter_executed',False),'dataset_accepted':False,'full_generator_training_started':False}
(ROOT/'receipts'/('strict_replay_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
expected='TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED' if a.mode=='witness' else 'TEXTOP_TWO_ROUND_FILTER_EXECUTION_FINISHED'
if code==0 and not timed_out and receipt.get('status')==expected:print('TEXTOP_STRICT_REPLAY_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
