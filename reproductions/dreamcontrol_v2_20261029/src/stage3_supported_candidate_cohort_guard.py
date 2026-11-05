"""Bounded first production cohort using the proven three-task DDPM sampler."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
p=argparse.ArgumentParser();p.add_argument('--task',choices=['pour','bimanual_box_lift','box_push_pull'],required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--cohort',type=pathlib.Path,required=True);a=p.parse_args()
assert 1<=a.count<=64 and a.start>=11000000;a.cohort.mkdir(parents=True,exist_ok=False)
status=a.cohort/'producer_guard.json';state={'status':'initializing','pid':os.getpid(),'candidate_count':a.count,'task':a.task,'start_id':a.start,'accepted_references':0};child=None;began=time.monotonic()
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(status)
try:
 spec=json.loads((D/'env_spec.json').read_text());assert spec['spec_hash']=='74af067a'
 cfg=json.loads((D/'config.json').read_text());env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(spec['spec']['module_paths'])};env.pop('CUDA_VISIBLE_DEVICES',None)
 source=B/'src/stage3_sample_supported29_batch.py';command=['nice','-n','10','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python','-u',str(source),'--task',a.task,'--count',str(a.count),'--start-index',str(a.start),'--output-dir',str(a.cohort/'candidates'),'--batch-size','2']
 assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2+100*1024**2
 log=(a.cohort/'producer.log').open('x');child=subprocess.Popen(command,env=env,cwd=cfg['trajgen'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='batch_candidates_sampling',child_pid=child.pid,command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest());save()
 while child.poll() is None:
  own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total,elapsed_seconds=time.monotonic()-began);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3 and total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-began<7200;time.sleep(3)
 assert child.wait()==0,'preserve actual prefix; never restart the same cohort'
 j=json.loads((a.cohort/'candidates/report.json').read_text());assert j['generated_candidates']==a.count
 state.update(status='batch_candidates_generated_postoptimization_pending',result=j,elapsed_seconds=time.monotonic()-began);save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
