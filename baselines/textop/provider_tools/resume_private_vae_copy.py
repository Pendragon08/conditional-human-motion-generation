"""Private immutable-byte SFTP resume, verified prefix, own bounded local worker."""
import argparse,hashlib,json,msvcrt,os,pathlib,shutil,subprocess,sys,time,traceback
from backup_filter_recovery_checkpoint import recorded_remote
from filter_recovery_windows_paths import guard_local
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
p=argparse.ArgumentParser();p.add_argument('--step',choices=['99700','100000'],required=True);p.add_argument('--run',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();assert a.run.isascii() and a.run.isidentifier()
E=O/a.run
def birth(process):
 import ctypes
 from ctypes import wintypes
 k=ctypes.WinDLL('kernel32',use_last_error=True);f=k.GetProcessTimes;f.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;f.restype=wintypes.BOOL
 values=[wintypes.FILETIME() for _ in range(4)];assert f(wintypes.HANDLE(int(process._handle)),*[ctypes.byref(x) for x in values])
 return (values[0].dwHighDateTime<<32)|values[0].dwLowDateTime

def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def digest(p,n=None):
 h=hashlib.sha256();remaining=n
 with p.open('rb') as f:
  while remaining is None or remaining>0:
   b=f.read(8*1024**2 if remaining is None else min(8*1024**2,remaining))
   if not b:break
   h.update(b)
   if remaining is not None:remaining-=len(b)
 return h.hexdigest()
if not a.worker:
 # An unfinished first client is never permission to start a duplicate.
 if (O/'r997/s_client_intent.json').exists() and not (O/'r997/s_client.json').exists():
  print(json.dumps(dict(status='EXISTING_PRIVATE_COPY_CLIENT_PENDING_NO_NEW_DISPATCH')));raise SystemExit(0)
 for intent in O.glob('*/client_intent.json'):
  outcome=intent.with_name('client_outcome.json');end=intent.with_name('worker_exit.json')
  if not outcome.exists() and not (end.exists() and json.loads(end.read_bytes()).get('client_reaped')):
   print(json.dumps(dict(status='EXISTING_RESUMABLE_CLIENT_UNKNOWN_NO_NEW_DISPATCH',intent=str(intent))));raise SystemExit(0)
 assert json.loads((O/'reget_fixture_v1/SUMMARY.json').read_bytes())['status']=='REAL_WINDOWS_SFTP_REGET_PREFIX_FIXTURE_PASSED'
 lock=(D/'filtered15000_vae_checkpoint_backup.lock').open('a+b');lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:print(json.dumps(dict(status='EXISTING_PRIVATE_COPY_LOCK_BUSY_NO_NEW_DISPATCH')));raise SystemExit(0)
 lock.close();E.mkdir(exist_ok=False)
 cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),'--step',a.step,'--run',a.run,'--worker']
 with (E/'worker.log').open('xb') as log:q=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS)
 result=dict(status='PRIVATE_RESUMABLE_COPY_WORKER_DISPATCHED_NOT_VERIFIED',worker_pid=q.pid,worker_start_filetime=birth(q),command=cmd,step=int(a.step),started_unix=time.time())
 write(E/'dispatch.json',result);print(json.dumps(result));raise SystemExit(0)
lock=None;client=None
try:
 write(E/'worker_started.json',dict(pid=os.getpid(),started_unix=time.time(),step=int(a.step)))
 lock=(D/'filtered15000_vae_checkpoint_backup.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 binding=None
 if a.step=='99700':
  ck=json.loads((B/'configs/user-vae100k-resume-spec.json').read_bytes())['resume_checkpoint'];dest=D/'checkpoints/private/filtered15000_vae200k_20261028/emergency_99700';final=dest/'emergency_99700.pth';receipt_root=O/'r997'
 else:
  j=json.loads((O/'user_vae100k_resume_train_v1/progress.json').read_bytes());assert j['closed_execution_verified'] and j['files']['receipt']['status']=='USER_VAE100000_RESUME_FINISHED'
  ck=j['files']['receipt']['selected_checkpoint'];binding=dict(kind='USER_VAE100000_RESUMED_FROM_CAPACITY_STOP',selected_step=100000,run='user_vae100k_resume_train_v1',receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit'])
  dest=D/'checkpoints/private/filtered15000_vae200k_20261028/step_100000';final=dest/'vae_state_100000.pth';receipt_root=O/'checkpoint100000_backup'
 guard_local(D,dest,directory=True);dest.mkdir(exist_ok=True)
 if final.exists():
  guard_local(D,final);assert final.stat().st_size==ck['bytes'] and digest(final)==ck['sha256']
  result=dict(status='REAL99700_EMERGENCY_PRIVATE_BACKUP_VERIFIED' if a.step=='99700' else 'USER_VAE100000_PRIVATE_BYTE_BACKUP_VERIFIED',checkpoint=ck,control_binding=binding,D_path=str(final),D_sha256=ck['sha256'],D_free_bytes=shutil.disk_usage(D).free,verified_existing_copy=True,transfer_dispatched=False,new_gpu_contexts=0,new_model_loads=0)
  receipt_root.mkdir(exist_ok=True)
  for p in [receipt_root/'backup.json',dest/'backup.json']:
   if not p.exists():write(p,result)
   else:assert json.loads(p.read_bytes())['D_sha256']==ck['sha256']
  write(E/'backup.json',result);write(E/'worker_exit.json',dict(status='ALREADY_EXACT_PRIVATE_COPY_PRESENT',actual_returncode=0));raise SystemExit(0)
 partial=dest/'vae_state.pth.partial';guard_local(D,partial)
 n=partial.stat().st_size if partial.exists() else 0;assert 0<=n<=ck['bytes']
 assert shutil.disk_usage(D).free>=2*1024**3+(ck['bytes']-n)+64*1024**2
 prefix=digest(partial) if n else hashlib.sha256(b'').hexdigest()
 code=r'''
import hashlib,json,os,pathlib,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from user_vae100k_resume_protocol import read,failed_source,completed_resume
from vae100k_control_protocol import verify_checkpoint
binding=BINDING
v=failed_source(read(r/'configs/user-vae100k-resume-spec.json')) if binding is None else completed_resume(binding)
assert v['checkpoint']==CHECKPOINT
proof=verify_checkpoint(v['checkpoint'],v['spec']['checkpoint_allowed_root']);left=PREFIX_BYTES;h=hashlib.sha256()
with pathlib.Path(v['checkpoint']['path']).open('rb') as f:
 while left:
  b=f.read(min(left,8*1024**2));assert b;left-=len(b);h.update(b)
print(json.dumps(dict(fingerprint=proof,prefix_sha256=h.hexdigest())))
'''.replace('BINDING',repr(binding)).replace('CHECKPOINT',repr(ck)).replace('PREFIX_BYTES',repr(n))
 before=json.loads(recorded_remote(code,E/'before_client.json',timeout=180));assert before['prefix_sha256']==prefix
 write(E/'intent.json',dict(checkpoint=ck,control_binding=binding,prefix_bytes=n,prefix_sha256=prefix,source_fingerprint=before['fingerprint'],new_gpu_contexts=0))
 cmd=['sftp','-b','-','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32']
 with (E/'transfer.stdout.txt').open('xb') as out,(E/'transfer.stderr.txt').open('xb') as err,(E/'resources.jsonl').open('x',encoding='utf-8') as resources:
  client=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=dest)
  write(E/'client_intent.json',dict(command=cmd,pid=client.pid,start_filetime=birth(client),timeout_seconds=21600,started_unix=time.time(),prefix_bytes=n))
  client.stdin.write(('reget '+ck['path']+' vae_state.pth.partial\n').encode());client.stdin.close();started=time.monotonic()
  while client.poll() is None:
   size=partial.stat().st_size if partial.exists() else 0;free=shutil.disk_usage(D).free
   resources.write(json.dumps(dict(unix=time.time(),bytes=size,free_bytes=free,client_pid=client.pid))+'\n');resources.flush()
   assert size<=ck['bytes'] and free>=2*1024**3+max(0,ck['bytes']-size)+16*1024**2 and time.monotonic()-started<21600
   try:client.wait(timeout=15)
   except subprocess.TimeoutExpired:pass
  outcome=dict(actual_client_exit_code=client.returncode,client_reaped=True,streams_final=True,transport='SFTP reget',prefix_bytes_preserved=n)
  write(E/'client_outcome.json',outcome);assert client.returncode==0
 guard_local(D,partial);assert partial.stat().st_size==ck['bytes'] and digest(partial)==ck['sha256']
 with partial.open('r+b') as f:f.flush();os.fsync(f.fileno())
 after=json.loads(recorded_remote(code,E/'after_client.json',timeout=180));assert after==before
 partial.rename(final);guard_local(D,final);assert shutil.disk_usage(D).free>=2*1024**3
 result=dict(status='REAL99700_EMERGENCY_PRIVATE_BACKUP_VERIFIED' if a.step=='99700' else 'USER_VAE100000_PRIVATE_BYTE_BACKUP_VERIFIED',checkpoint=ck,control_binding=binding,D_path=str(final),D_sha256=ck['sha256'],D_free_bytes=shutil.disk_usage(D).free,source_fingerprint_stable=True,actual_sftp_exit_code=0,resumed_prefix_bytes=n,new_gpu_contexts=0,new_model_loads=0)
 receipt_root.mkdir(exist_ok=True);write(receipt_root/'backup.json',result);write(dest/'backup.json',result);write(E/'backup.json',result);write(E/'worker_exit.json',dict(status='PRIVATE_BYTE_BACKUP_VERIFIED',actual_returncode=0));print(json.dumps(result))
except BaseException as error:
 if isinstance(error,SystemExit):raise
 if client is not None and client.poll() is None:client.kill();client.wait(timeout=15)
 write(E/'worker_exit.json',dict(status='PRIVATE_COPY_FAILED_PARTIAL_RETAINED',actual_returncode=1,error=repr(error),traceback=traceback.format_exc(),client_reaped=client is None or client.poll() is not None));raise
finally:
 if lock is not None:lock.close()
