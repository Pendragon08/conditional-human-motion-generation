"""One bounded private generator state diagnostic, with actual exit evidence."""
import argparse,fcntl,json,os,pathlib,re,signal,subprocess,sys,threading,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
lock=(ROOT/'receipts/generator_state_diagnostic.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=json.loads((ROOT/'configs/generator-state-witness.json').read_text());out=ROOT/'generator_validation'/a.run
assert not out.exists() and spec['dispatch_enabled']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
env['PYTHONPATH']=str(ROOT/'scripts')+os.pathsep+env.get('PYTHONPATH','')
cmd=[str(ROOT/'env/bin/python'),str(ROOT/'scripts/generator_state_witness.py'),'--spec',str(ROOT/'configs/generator-state-witness.json'),
 '--base-spec',str(ROOT/'configs/paper69-env-spec.json'),'--plan',str(ROOT/'configs/paper-step2.json'),'--output',str(out)]
started=time.monotonic();samples=[];done=threading.Event();timed_out=False
def sample():
 while not done.is_set():
  try:
   result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
   samples.append({'elapsed_seconds':time.monotonic()-started,'device_memory_mib_including_other_processes':result.stdout.strip()})
  except subprocess.TimeoutExpired:samples.append({'elapsed_seconds':time.monotonic()-started,'sampling_error':'nvidia-smi timed out'})
  done.wait(2)
watch=threading.Thread(target=sample,daemon=True);watch.start()
with (ROOT/'logs'/('generator_state_'+a.run+'.log')).open('xb') as target:
 child=subprocess.Popen(cmd,env=env,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
  preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
 (ROOT/'receipts'/('generator_state_'+a.run+'_launch.json')).write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'command':cmd},indent=2))
 try:code=child.wait(timeout=spec['supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
done.set();watch.join(timeout=12)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,
 'numerical_status':receipt.get('status'),'command':cmd,'device_memory_samples':samples,'optimizer_updates':receipt.get('optimizer_updates'),
 'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
(ROOT/'receipts'/('generator_state_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='device_memory_samples'}),flush=True)
if code==0 and not timed_out and receipt.get('status')=='TEXTOP_GENERATOR_STATE_WITNESS_PASSED':print('TEXTOP_GENERATOR_STATE_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
