"""Bounded native trainer dispatch. No old adapter/PPO restart or foreign signals."""
import argparse,ctypes,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';W=D/'native_universal_20261104';S=R/'stage3_pick29_20261103'
sys.path.insert(0,str(D/'src'));sys.path.insert(0,str(S/'src'))
from stage3_pick_parallel_controller import resources,stop_group
from stage3_pick500_replacement_queue import snapshot
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['resume'],required=True);p.add_argument('--label',required=True);a=p.parse_args()
assert Path(a.label).name==a.label and a.label.startswith(('resume_',))
out=W/a.label;assert not out.exists() and not (W/(a.label+'_guard.json')).exists()
spec=json.loads((W/'environment_spec.json').read_text())['spec'];cache=Path('/dev/shm/dcv2_native_smoke_author_v1');assert cache.is_dir() and cache.stat().st_uid==os.getuid()
assert shutil.disk_usage('/dev/shm').free>2*1024**3+spec['cache_cap_bytes']
env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PXR_WORK_THREAD_LIMIT':'1','MAX_JOBS':'1',
 'TORCHINDUCTOR_COMPILE_THREADS':'1','NVRTC_DISABLE_PCH':'1','CUDA_CACHE_DISABLE':'1','HOME':str(cache),
 'XDG_CACHE_HOME':str(cache/'xdg'),'WARP_CACHE_PATH':str(cache/'warp'),'TMPDIR':str(cache),'TMP':str(cache),'TEMP':str(cache),
 'TRITON_CACHE_DIR':str(cache/'triton'),'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'),'MPLCONFIGDIR':str(cache/'mpl')}
report={'mode':a.mode,'label':a.label,'pid':os.getpid(),'status':'preflight','cache':str(cache),'scope':'bounded native witness, not formal training'}
def save():
 p=W/(a.label+'_guard.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(p)
child=None;start=time.monotonic();expected='native_capacity_passed' if a.mode=='capacity' else 'native_resume_passed'
try:
 assert shutil.disk_usage('/data').free>=spec['runtime_disk_floor']+spec['output_cap_bytes']
 cmd=[spec['python'],'-I','-B','-u',str(D/'src/native_tracker500_resume_worker.py'),'--mode',a.mode,'--output',str(out)]
 log=(W/(a.label+'.log')).open('x');child=subprocess.Popen(cmd,cwd=str(W/'source'),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status='running',child_pid=child.pid,command=cmd);save();finished=None
 while child.poll() is None:
  own=resources(child.pid);all_owned=snapshot();cache_size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
  report.update(resources=own,aggregate_resources=all_owned,cache_bytes=cache_size,elapsed_seconds=time.monotonic()-start);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
  assert all_owned['owned_gpu_bytes']<=24*1024**3 and all_owned['owned_rss_bytes']<=64*1024**3
  assert cache_size<=spec['cache_cap_bytes'] and shutil.disk_usage('/data').free>=spec['runtime_disk_floor']
  assert time.monotonic()-start<(90 if a.mode=='kernel' else 1800)
  if (out/'status.json').exists():
   s=json.loads((out/'status.json').read_text())
   if s['status']=='failed_preserved':raise RuntimeError(s.get('error'))
   if s['status']==expected:
    if finished is None:finished=time.monotonic()
    if time.monotonic()-finished>30:stop_group(child.pid);break
  time.sleep(2)
 result=json.loads((out/'status.json').read_text());assert result['status']==expected
 report.update(status='passed',worker_result=result,returncode=child.poll());save();print(json.dumps(report),flush=True)
except BaseException as e:
 report.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child:stop_group(child.pid)
 raise
