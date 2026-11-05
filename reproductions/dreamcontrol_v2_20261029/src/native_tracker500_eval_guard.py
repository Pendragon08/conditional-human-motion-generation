"""Guard only the new read-only evaluator; no PPO or foreign job signals."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';W=N/'evaluation_entry500_v1'
sys.path[:0]=[str(D/'src'),str(R/'stage3_pick29_20261103/src')]
from stage3_pick_parallel_controller import resources,stop_group
from stage3_pick500_replacement_queue import snapshot
out=N/'evaluation_entry500_v1_guard.json';assert not out.exists() and not W.exists()
spec=json.loads((N/'environment_spec.json').read_text())['spec'];cache=Path('/dev/shm/dcv2_native_smoke_author_v1')
assert cache.is_dir() and cache.stat().st_uid==os.getuid()
env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PXR_WORK_THREAD_LIMIT':'1','MAX_JOBS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1','NVRTC_DISABLE_PCH':'1','CUDA_CACHE_DISABLE':'1','HOME':str(cache),'XDG_CACHE_HOME':str(cache/'xdg'),'WARP_CACHE_PATH':str(cache/'warp'),'TMPDIR':str(cache),'TMP':str(cache),'TEMP':str(cache),'TRITON_CACHE_DIR':str(cache/'triton'),'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'),'MPLCONFIGDIR':str(cache/'mpl')}
state={'status':'preflight','pid':os.getpid()};child=None;started=time.monotonic()
def save():
 t=out.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(out)
try:
 assert shutil.disk_usage('/data').free>spec['runtime_disk_floor']+100*1024**2
 assert json.loads((N/'benchmark_8912_v1_guard.json').read_text())['status']=='passed'
 log=(N/'evaluation_entry500_v1.log').open('x')
 child=subprocess.Popen([spec['python'],'-I','-B','-u',str(D/'src/native_tracker500_eval_worker.py')],cwd=str(N/'source'),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='running',child_pid=child.pid);save();finished=None
 while child.poll() is None:
  own=resources(child.pid);all_owned=snapshot();state.update(resources=own,aggregate_resources=all_owned,elapsed_seconds=time.monotonic()-started);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
  assert all_owned['owned_gpu_bytes']<=24*1024**3 and all_owned['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=spec['runtime_disk_floor'] and time.monotonic()-started<1800
  if (W/'status.json').exists():
   j=json.loads((W/'status.json').read_text())
   if j['status']=='failed_preserved':raise RuntimeError(j['error'])
   if j['status']=='native_evaluation_entry_passed':
    if finished is None:finished=time.monotonic()
    if time.monotonic()-finished>30:stop_group(child.pid);break
  time.sleep(2)
 j=json.loads((W/'status.json').read_text());assert j['status']=='native_evaluation_entry_passed'
 state.update(status='passed',result=j,returncode=child.poll());save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child:stop_group(child.pid)
 raise
