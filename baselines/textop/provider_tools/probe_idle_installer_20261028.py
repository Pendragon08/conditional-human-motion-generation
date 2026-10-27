"""Read-only installer proof; distinguish OS credential daemons from trainers."""
import json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];E=B/'results/storage_user_cleanup_20261028'
CODE=r'''
import hashlib,json,os,pathlib,stat,time,shutil
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
p=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/cache/downloads/IsaacGym_Preview_4.tar.gz')
assert p.resolve()==p and p.stat().st_uid==1013 and p.stat().st_size==201141673 and p.stat().st_nlink==1
for q in [p,p.parent,p.parent.parent,p.parent.parent.parent]:assert not q.is_symlink() and q.stat().st_uid==1013
finger=lambda s:dict(inode=s.st_ino,device=s.st_dev,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,owner_uid=s.st_uid)
before=finger(p.stat());h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert finger(p.stat())==before
references=[];daemon_limits=[];readable=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=1013:continue
  cmd=(proc/'cmdline').read_bytes();assert str(p).encode() not in cmd,'An installer command is active'
  try:fds=list((proc/'fd').iterdir())
  except PermissionError:
   name=(proc/'comm').read_text().strip()
   assert name in ['(sd-pam)','ssh-agent','gpg-agent','gnome-keyring-d','sshd'],(proc.name,name)
   daemon_limits.append(dict(pid=int(proc.name),name=name,fd_access=False));continue
  readable.append(int(proc.name))
  for fd in fds:
   try:
    if os.readlink(fd)==str(p):references.append(dict(pid=int(proc.name),fd=fd.name))
   except (FileNotFoundError,ProcessLookupError):pass
 except (FileNotFoundError,ProcessLookupError):pass
assert not references and 165381 in readable and 165374 in readable
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert hashlib.sha256((r/'configs/filtered15000-vae-spec.json').read_bytes()).hexdigest()=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
for pid,start in [(165374,'1432055921'),(165381,'1432055933')]:
 proc=pathlib.Path('/proc')/str(pid);fields=(proc/'stat').read_text().split(') ',1)[1].split();assert fields[19]==start
 assert b'filtered15000_vae200k_train_v1' in (proc/'cmdline').read_bytes()
sdk=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym');assert sdk.is_dir()
print(json.dumps(dict(status='IDLE_SDK_INSTALLER_SOURCE_VERIFIED',path=str(p),source_fingerprint=before,sha256=h.hexdigest(),
 readable_owned_processes_count=len(readable),installer_references=references,non_training_daemon_fd_limitations=daemon_limits,
 all_trainer_fds_readable=True,global_fd_visibility_complete=not bool(daemon_limits),extracted_sdk_directory_retained=str(sdk),
 source_file_deleted=False,training_source_modified=False,new_gpu_contexts=0,new_model_loads=0,observed_unix=time.time(),data_free_bytes=shutil.disk_usage(p).free)))
'''
if __name__=='__main__':
 j=json.loads(recorded_remote(CODE,E/'server_installer_verify_v2_client.json',timeout=90))
 with (E/'server_installer_source_verified.json').open('x',encoding='utf-8') as f:json.dump(j,f,indent=2)
 print(json.dumps(j))
