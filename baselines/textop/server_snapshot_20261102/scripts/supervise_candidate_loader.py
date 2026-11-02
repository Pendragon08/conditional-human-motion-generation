"""Finite candidate build or loader witness; no generator training launch."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--build',action='store_true');a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
out=ROOT/'generator_validation'/a.run;assert not out.exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
env['PYTHONPATH']=str(ROOT/'scripts')+os.pathsep+env.get('PYTHONPATH','')
cmd=[str(ROOT/'env/bin/python'),str(ROOT/'scripts/paper69_candidate_loader.py'),'--spec',str(ROOT/'configs/candidate-loader-spec.json'),
 '--base-spec',str(ROOT/'configs/paper69-env-spec.json'),'--output',str(out)]
if a.build:cmd.append('--build')
start=time.monotonic();timed_out=False
with (ROOT/'logs'/('candidate_loader_'+a.run+'.log')).open('xb') as target:
 child=subprocess.Popen(cmd,env=env,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
   preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
 (ROOT/'receipts'/('candidate_loader_'+a.run+'_launch.json')).write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'command':cmd},indent=2))
 try:code=child.wait(timeout=5400 if a.build else 1800)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,
 'numerical_status':receipt.get('status'),'command':cmd,'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
(ROOT/'receipts'/('candidate_loader_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if code==0 and not timed_out and receipt.get('status')=='TEXTOP_CANDIDATE69_LOADER_WITNESS_PASSED':print('TEXTOP_CANDIDATE_LOADER_SUPERVISED_EXIT0',flush=True)
else:sys.exit(1)
