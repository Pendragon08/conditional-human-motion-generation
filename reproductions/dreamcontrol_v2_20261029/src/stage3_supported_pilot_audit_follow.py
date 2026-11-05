import json,os,pathlib,subprocess,sys,time,traceback,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'pilot_v2';S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
out=W/'audit_follow_status.json';assert not out.exists();state={'status':'waiting_three_task_sampling','pid':os.getpid()};child=None;began=time.monotonic()
def save():
 t=out.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(out)
try:
 save()
 while True:
  j=json.loads((W/'status.json').read_text())
  if j['status']=='failed_preserved':raise RuntimeError('sampling failed; audit not launched')
  if j['status']=='three_task_pilot_candidates_generated_postoptimization_pending':break
  assert time.monotonic()-began<6000;time.sleep(10)
 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'};env.pop('CUDA_VISIBLE_DEVICES',None)
 log=(W/'candidate_audit.log').open('x')
 child=subprocess.Popen(['nice','-n','10','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python','-u',str(B/'src/stage3_supported_pilot_audit.py')],env=env,cwd=str(B),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='independent_candidate_audit_running',child_pid=child.pid);save();audit_began=time.monotonic()
 while child.poll() is None:
  a=resources(child.pid);total=snapshot();assert a['gpu_bytes']<=12*1024**3 and a['rss_bytes']<=32*1024**3
  assert total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-audit_began<300
  time.sleep(2)
 assert child.wait()==0;report=json.loads((W/'candidate_audit.json').read_text());assert report['candidate_count']==6 and report['accepted_reference_count']==0
 state.update(status='three_task_pilot_candidate_audit_passed_postoptimization_pending',audit_report=str(W/'candidate_audit.json'));save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
