"""Resume the verified BMLrub upload after the old GMR mutual-exclusion guard.

Does not upload again, restart a motion batch, delete data or control other jobs.
Wait for validated parallel activation, launch once, then preserve normal backup.
"""
import datetime,gzip,json,msvcrt,pathlib,shlex,shutil,subprocess,sys,tarfile,time
from session import REPO,SSH,remote
from restore_indexed_cache import sha
from continue_remaining_indexed import load
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024';RAM='/dev/shm/ouyanghao_textop_20261025/bmlrub_indexed_v1'
PY='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
dest=BASE/'results/data_preparation_20261025';state_path=dest/'bmlrub_indexed_queue.json'
sys.stdout.reconfigure(encoding='utf-8')
def state(status,**values):
 old=load(state_path)
 if 'error' in old:old['previous_error']=old.pop('error')
 old.update(status=status,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
 temp=state_path.with_suffix('.json.partial');temp.write_text(json.dumps(old,indent=2),encoding='utf-8');temp.replace(state_path)
 print(json.dumps({'status':status,**values}),flush=True)
def publish():
 for name in ['collect_data_preparation.py','collect_tracker_progress.py','publish.py']:
  p=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
  if p.returncode:
   (dest/'bmlrub_observer_publication_error.json').write_text(json.dumps({'script':name,'exit':p.returncode,'error':p.stderr[-2000:]}),encoding='utf-8')
   break
def main():
 lock=(dest/'bmlrub_resume_observer.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 old=load(state_path);assert old['status']=='STOPPED_WITH_ERROR_ALL_DATA_RETAINED' and 'Another GMR is active' in old['error']
 history=dest/'queue_history';history.mkdir(exist_ok=True);snapshot=history/'bmlrub_mutual_exclusion_20261025.json';assert not snapshot.exists();snapshot.write_text(json.dumps(old,indent=2),encoding='utf-8')
 state('WAITING_FOR_VALIDATED_PARALLEL_DISPATCH',resume_kind='reuse fully staged and hashed raw inputs; no duplicate transfer',physical_filter_executed=False,dataset_accepted=False,full_training_started=False)
 for _ in range(360):
  if load(BASE/'configs/gmr-parallel-activation.json').get('mode')=='PARALLEL_VALIDATED':break
  time.sleep(10)
 else:raise TimeoutError('Parallel witness not activated within1h; staged assets preserved')
 source_hash=sha(BASE/'scripts/prepare_paper_data.py')
 launch=json.loads(remote(f'''
import pathlib,json,hashlib,subprocess,os,time,shutil
r=pathlib.Path({ROOT!r});ram=pathlib.Path({RAM!r});assert os.getuid()==1013
assert not (r/'receipts/bmlrub_indexed_launch.json').exists()
rows=json.loads((ram/'pairs.private.json').read_text());assert len(rows)==2139
assert len({{x['canonical_source'] for x in rows}})==2139
for row in rows:
 p=pathlib.Path(row['raw_stageii_candidates'][0]);assert p.parent==ram/'raw' and p.stat().st_uid==1013
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
assert hashlib.sha256((r/'scripts/prepare_paper_data.py').read_bytes()).hexdigest()=={source_hash!r}
assert json.loads((r/'configs/gmr-parallel-activation.json').read_text())['mode']=='PARALLEL_VALIDATED'
assert shutil.disk_usage('/data').free>2*1024**3 and shutil.disk_usage('/dev/shm').free>8*1024**3
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();owner=p.stat().st_uid
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not(owner==1013 and 'scripts/prepare_paper_data.py' in cmd and '--output' in cmd),'Another GMR remains active'
output=ram/'prepared_v1';assert not output.exists(),'Refusing to overwrite any earlier preprocessing attempt'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[{PY!r},str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(ram/'pairs.private.json'),'--output',str(output)]
with (r/'logs/data_prep_bmlrub_parallel_v1.log').open('xb') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{{0,1,2,3}}))
proof={{'pid':p.pid,'command':cmd,'cpu_only':True,'parallel_workers':4,'verified_raw':2139,'source_sha256':{source_hash!r},'time_utc':time.strftime('%FT%TZ',time.gmtime()),'resume_after_mutual_exclusion':True}}
(r/'receipts/bmlrub_indexed_launch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
''',timeout=180))
 state('RUNNING_GMR_PARALLEL_CPU_ONLY',launch=launch);last=-1
 for _ in range(8640):
  try:progress=json.loads(remote("import pathlib;p=pathlib.Path('"+RAM+"/prepared_v1/progress.json');print(p.read_text() if p.exists() else '{}')"))
  except json.JSONDecodeError:time.sleep(1);continue
  if progress.get('processed',0)//100>last:state('RUNNING_GMR_PARALLEL_CPU_ONLY',progress=progress);publish();last=progress.get('processed',0)//100
  if progress.get('status') in ['PREPROCESSING_FINISHED_FILTER_PENDING','PREPROCESSING_FAILED']:break
  time.sleep(30)
 else:raise TimeoutError('Preprocessing exceeded72h; all inputs and processes preserved')
 assert progress['status']=='PREPROCESSING_FINISHED_FILTER_PENDING' and progress['completed_sequences']==2139 and progress['failed_sequences']==0
 estimate=int(remote("import pathlib;p=pathlib.Path('"+RAM+"/prepared_v1');print(sum(x.stat().st_size for x in p.rglob('*') if x.is_file()))"))
 for _ in range(8640):
  if shutil.disk_usage(LOCAL).free>estimate*1.05+2*1024**3:break
  state('WAITING_FOR_DURABLE_BACKUP_SPACE',required_bytes=int(estimate*1.05+2*1024**3),progress=progress);time.sleep(30)
 else:raise RuntimeError('Backup space unavailable; outputs preserved')
 state('BACKING_UP_PRIVATE_PREPARED_OUTPUT',progress=progress)
 target=LOCAL/'datasets/paper29_bmlrub_20261025.private.tar.gz';partial=target.with_suffix('.gz.partial');assert not target.exists() and not partial.exists()
 with partial.open('xb') as output:
  proc=subprocess.run([*SSH,'tar -C '+shlex.quote(RAM)+' -czf - prepared_v1 pairs.private.json'],stdout=output,stderr=subprocess.PIPE,timeout=28800)
 assert proc.returncode==0,proc.stderr[-2000:]
 with gzip.open(partial,'rb') as f:
  while f.read(8*1024**2):pass
 with tarfile.open(partial) as archive:assert sum(x.name.endswith('.private.npz') for x in archive.getmembers())==2139
 digest=sha(partial);partial.rename(target)
 state('PREPROCESSING_BACKED_UP_FILTER_PENDING',progress=progress,archive_sha256=digest,archive_bytes=target.stat().st_size);publish()
if __name__=='__main__':
 try:main()
 except BaseException as e:state('STOPPED_WITH_ERROR_ALL_DATA_RETAINED',error=repr(e));raise
