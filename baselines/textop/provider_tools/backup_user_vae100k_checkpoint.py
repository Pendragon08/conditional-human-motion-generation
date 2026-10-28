"""Private byte copy of the exact user100k prefix after proven original stop."""
import ast,hashlib,json,msvcrt,os,pathlib,shutil,time
from backup_filter_recovery_checkpoint import recorded_client,recorded_remote
from filter_recovery_windows_paths import guard_local
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def main():
 rp=O/'user_vae100k_resume_train_v1/progress.json'
 if rp.exists():
  j=json.loads(rp.read_bytes());assert j['kind']=='resume' and j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
  r=j['files']['receipt'];assert r['status']=='USER_VAE100000_RESUME_FINISHED'
  binding=dict(kind='USER_VAE100000_RESUMED_FROM_CAPACITY_STOP',selected_step=100000,run='user_vae100k_resume_train_v1',receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit'])
 else:
  j=json.loads((O/'user_vae100k_control_v1/progress.json').read_bytes());assert j['kind']=='control' and j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
  r=j['files']['receipt'];assert r['status']=='USER_VAE100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
  binding=dict(kind='USER_REQUESTED_VAE100000_PREFIX',selected_step=100000,control_run='user_vae100k_control_v1',control_receipt_sha256=j['sha256']['receipt'],control_exit_sha256=j['sha256']['exit'])
 ck=r['selected_checkpoint'];assert ck['manager_step']==100000
 dest=D/'checkpoints/private/filtered15000_vae200k_20261028/step_100000'
 guard_local(D,dest,directory=True,must_absent=True)
 assert shutil.disk_usage(D).free>=2*1024**3+ck['bytes']+8*1024**2
 lockpath=D/'filtered15000_vae_checkpoint_backup.lock';guard_local(D,lockpath)
 lock=lockpath.open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 E=O/'checkpoint100000_backup';E.mkdir(exist_ok=False)
 code=r'''
import os,pathlib,json,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from user_vae100k_completion import stopped_vae100k
from vae100k_control_protocol import verify_checkpoint
v=stopped_vae100k(BINDING);assert v['checkpoint']==CHECKPOINT
age=time.time()-pathlib.Path(v['checkpoint']['path']).stat().st_mtime
if age<31:time.sleep(31-age)
p=verify_checkpoint(v['checkpoint'],v['spec']['checkpoint_allowed_root']);print(json.dumps(p))
'''.replace('BINDING',repr(binding)).replace('CHECKPOINT',repr(ck))
 ast.parse(code);before=json.loads(recorded_remote(code,E/'source_before_client.json',timeout=90));write(E/'source_before.json',before)
 dest.mkdir();guard_local(D,dest,directory=True)
 intent=dict(status='USER_VAE100K_PRIVATE_COPY_INTENT',checkpoint=ck,control_binding=binding,source_fingerprint=before,tool_sha256=sha(__file__))
 write(E/'intent.json',intent);write(dest/'intent.json',intent)
 partial=dest/'vae_state.pth.partial';final=dest/'vae_state_100000.pth';guard_local(D,partial,must_absent=True);guard_local(D,final,must_absent=True)
 cmd=['scp','-B','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+ck['path'],str(partial)]
 client=recorded_client('',E/'scp_client.json',timeout=1200,command=cmd)
 assert client['actual_client_exit_code']==0 and client['client_reaped'] and client['streams_final'] and not client['timed_out']
 guard_local(D,partial);assert partial.stat().st_size==ck['bytes'] and sha(partial)==ck['sha256']
 with partial.open('r+b') as f:f.flush();os.fsync(f.fileno())
 after=json.loads(recorded_remote(code,E/'source_after_client.json',timeout=90));assert after==before;write(E/'source_after.json',after)
 assert shutil.disk_usage(D).free>=2*1024**3+8*1024**2
 partial.rename(final);guard_local(D,final)
 result=dict(status='USER_VAE100000_PRIVATE_BYTE_BACKUP_VERIFIED',checkpoint=ck,control_binding=binding,D_path=str(final),D_sha256=sha(final),D_free_bytes=shutil.disk_usage(D).free,
  source_before_sha256=sha(E/'source_before.json'),source_after_sha256=sha(E/'source_after.json'),actual_scp_exit_code=0,source_fingerprint_stable=True,
  checkpoint_payload_deserialized=False,new_model_loads=0,new_gpu_contexts=0,signals_sent=0,files_deleted=0)
 assert result['D_sha256']==ck['sha256'] and result['D_free_bytes']>=2*1024**3
 write(E/'backup.json',result);write(dest/'backup.json',result);print(json.dumps(result))
if __name__=='__main__':main()
