"""Apply only published exact-part cleanup and a verified private installer move."""
import argparse,gzip,hashlib,json,os,pathlib,shutil,subprocess,time
from backup_filter_recovery_checkpoint import recorded_client,recorded_remote
from filter_recovery_windows_paths import guard_local
from session import SSH
B=pathlib.Path(__file__).resolve().parents[1];R=B.parents[1]
E=B/'results/storage_user_cleanup_20261028';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def new(path,value):
 with pathlib.Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['D','copy-sdk','server'],required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 assert len(a.commit)==40 and all(c in '0123456789abcdef' for c in a.commit)
 for n in ['provider_tools/remove_verified_transport_parts_20261028.ps1','provider_tools/execute_user_cleanup_20261028.py',
  'provider_tools/restore_contact_transport_part_20261028.py','provider_tools/probe_idle_installer_20261028.py',
  'results/storage_user_cleanup_20261028/D_recovery_verified.json','results/storage_user_cleanup_20261028/server_installer_source_verified.json']:
  v=recorded_client('',E/(a.stage+'_git_'+pathlib.Path(n).stem+'.json'),timeout=30,
   command=['git','-C',str(R),'show',a.commit+':baselines/textop/'+n])
  assert v['actual_client_exit_code']==0 and hashlib.sha256(v['stdout'].encode('utf-8')).hexdigest()==sha(B/n),'Published code/proof bytes must match'
 env=os.environ.copy();env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
 v=recorded_client('',E/(a.stage+'_remote_commit_client.json'),timeout=90,
  command=['git','-C',str(R),'ls-remote','origin','refs/heads/codex/textop-reproduction-20261024'],env=env)
 assert v['actual_client_exit_code']==0 and v['stdout'].split()[0]==a.commit
 source=json.loads((E/'server_installer_source_verified.json').read_text(encoding='utf-8'))
 if a.stage=='D':
  native=pathlib.Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0'
  env=os.environ.copy();env['PSModulePath']=str(native/'Modules')
  v=recorded_client('',E/'D_apply_client.json',timeout=240,env=env,command=[str(native/'powershell.exe'),'-NoProfile','-NonInteractive','-File',str(B/'provider_tools/remove_verified_transport_parts_20261028.ps1'),'-Proof',str(E/'D_recovery_verified.json'),'-Evidence',str(E),'-BackedUpCommit',a.commit])
  assert v['actual_client_exit_code']==0 and v['client_reaped'] and v['streams_final'] and not v['timed_out'],str(E/'D_apply_client.json')
  print(v['stdout']);return
 backup=D/'software_recovery_private_20261028'
 if a.stage=='copy-sdk':
  assert json.loads((E/'D_cleanup.json').read_text())['status']=='VERIFIED_REDUNDANT_TRANSPORT_PARTS_REMOVED'
  guard_local(D,backup,directory=True,must_absent=True)
  required=2*1024**3+3*553648128+source['source_fingerprint']['bytes']+8*1024**2
  assert shutil.disk_usage(D).free>=required
  backup.mkdir();guard_local(D,backup,directory=True)
  partial=backup/'IsaacGym_Preview_4.tar.gz.partial';final=backup/'IsaacGym_Preview_4.tar.gz'
  guard_local(D,partial,must_absent=True);guard_local(D,final,must_absent=True)
  new(backup/'intent.json',dict(source=source,code_commit=a.commit,required_D_bytes_including_remaining_VAE_slots=required))
  cmd=['scp','-B','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+source['path'],str(partial)]
  v=recorded_client('',E/'installer_scp_client.json',timeout=900,command=cmd)
  assert v['actual_client_exit_code']==0 and v['client_reaped'] and v['streams_final'] and not v['timed_out']
  guard_local(D,partial);assert partial.stat().st_size==source['source_fingerprint']['bytes'] and sha(partial)==source['sha256']
  total=0
  with gzip.open(partial,'rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):total+=len(block)
  with partial.open('r+b') as f:f.flush();os.fsync(f.fileno())
  partial.rename(final);guard_local(D,final)
  result=dict(status='PRIVATE_INSTALLER_BYTE_BACKUP_VERIFIED',path=str(final),bytes=final.stat().st_size,sha256=source['sha256'],gzip_full_CRC_verified=True,
   decompressed_bytes=total,code_commit=a.commit,source=source,D_free_bytes=shutil.disk_usage(D).free,D_required_for_remaining_VAE_slots=2*1024**3+3*553648128,
   proprietary_installer_uploaded_publicly=False,new_gpu_contexts=0)
  assert result['D_free_bytes']>=result['D_required_for_remaining_VAE_slots']
  new(backup/'backup.json',result);new(E/'installer_private_backup.json',result);print(json.dumps(result));return
 copied=json.loads((E/'installer_private_backup.json').read_text(encoding='utf-8'))
 assert copied['status']=='PRIVATE_INSTALLER_BYTE_BACKUP_VERIFIED'
 local=pathlib.Path(copied['path']);guard_local(D,local);assert sha(local)==source['sha256']
 import probe_idle_installer_20261028 as probe
 # Same bounded read-only probe runs inside the application phase before unlink.
 code=probe.CODE.rsplit('print(json.dumps',1)[0]
 code+='''
expected=EXPECTED;commit=COMMIT;backup_sha=BACKUP_SHA
assert finger(p.stat())==expected['source_fingerprint'] and h.hexdigest()==expected['sha256']
out=r/'receipts/storage_user_cleanup_20261028_v1';assert not out.exists();out.mkdir(mode=0o700)
sdk_before=dict(inode=sdk.stat().st_ino,mtime_ns=sdk.stat().st_mtime_ns)
before_free=shutil.disk_usage(p).free
intent=dict(status='PRIVATE_BACKED_UP_EXACT_INSTALLER_MOVE_INTENT',source=expected,private_D_backup_receipt_sha256=backup_sha,code_commit=commit,
 global_fd_visibility_complete=not bool(daemon_limits),non_training_daemon_fd_limitations=daemon_limits)
with (out/'intent.json').open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
assert p.parent==pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/cache/downloads') and finger(p.stat())==expected['source_fingerprint']
p.unlink()
assert not p.exists() and sdk_before==dict(inode=sdk.stat().st_ino,mtime_ns=sdk.stat().st_mtime_ns)
assert hashlib.sha256((r/'configs/filtered15000-vae-spec.json').read_bytes()).hexdigest()=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
result=dict(status='OLD_SOFTWARE_INSTALLER_MOVED_TO_PRIVATE_D_VERIFIED',removed_server_installer=str(p),logical_bytes_relocated=expected['source_fingerprint']['bytes'],
 data_free_before_bytes=before_free,data_free_after_bytes=shutil.disk_usage(r).free,private_D_backup_receipt_sha256=backup_sha,code_commit=commit,
 extracted_sdk_unchanged=True,active_VAE_source_spec_unchanged=True,new_gpu_contexts=0,signals_sent=False,datasets_removed=0,checkpoints_removed=0,unique_records_removed=0,
 global_fd_visibility_complete=not bool(daemon_limits),non_training_daemon_fd_limitations=daemon_limits)
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''
 code=code.replace('EXPECTED',repr(source)).replace('COMMIT',repr(a.commit)).replace('BACKUP_SHA',repr(sha(E/'installer_private_backup.json')))
 import ast;ast.parse(code)
 result=json.loads(recorded_remote(code,E/'server_apply_client.json',timeout=90));new(E/'server_cleanup.json',result);print(json.dumps(result))
if __name__=='__main__':main()
