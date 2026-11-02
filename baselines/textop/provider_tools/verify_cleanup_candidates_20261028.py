"""Verify exact duplicate transport parts and an idle private SDK installer."""
import gzip,hashlib,json,os,pathlib,sys,tarfile,time
from backup_filter_recovery_checkpoint import recorded_remote
from filter_recovery_windows_paths import guard_local
B=pathlib.Path(__file__).resolve().parents[1]
E=B/'results/storage_user_cleanup_20261028';E.mkdir(exist_ok=False)
F=pathlib.Path('D:/具身智能/hab-contact-aware-world-model/release')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
root=pathlib.Path('D:/具身智能');guard_local(root,F,directory=True)
metadata=json.loads((F/'contact-v1.release.json').read_text(encoding='utf-8'))
manraw=(F/metadata['manifest']).read_bytes();manifest=json.loads(manraw)
assert hashlib.sha256(manraw).hexdigest()==metadata['manifest_sha256']
ghpath=B.parents[3]/'textop_reproduction_20261024/cleanup_user_authorized_20261028_v1/github_hab_contact_metadata.json'
# Resolve the task sibling explicitly, independent of the repo's depth.
ghpath=B.parents[2]/'textop_reproduction_20261024/cleanup_user_authorized_20261028_v1/github_hab_contact_metadata.json'
if not ghpath.exists():
 ghpath=pathlib.Path('work/textop_reproduction_20261024/cleanup_user_authorized_20261028_v1/github_hab_contact_metadata.json').resolve()
release=json.loads(ghpath.read_text(encoding='utf-8'))[0]
assets={x['name']:x for x in release['assets']}
for name,expected,size in [('contact-v1.manifest.json',metadata['manifest_sha256'],len(manraw)),('contact-v1.release.json',sha(F/'contact-v1.release.json'),(F/'contact-v1.release.json').stat().st_size)]:
 assert assets[name]['digest']=='sha256:'+expected and assets[name]['size']==size and assets[name]['state']=='uploaded'
full=F/metadata['archive'];guard_local(root,full)
st=full.stat();assert st.st_size==metadata['archive_bytes']
whole=hashlib.sha256();rows=[];offset=0
with full.open('rb') as archive:
 for row in metadata['parts']:
  target=F/row['name'];guard_local(root,target);partst=target.stat()
  assert partst.st_size==row['bytes'] and assets[row['name']]['state']=='uploaded' and assets[row['name']]['size']==row['bytes']
  assert assets[row['name']]['digest']=='sha256:'+row['sha256']
  ph=hashlib.sha256();regenerated=hashlib.sha256();left=row['bytes']
  with target.open('rb') as p:
   while left:
    a=archive.read(min(left,8*1024**2));b=p.read(len(a));assert a and a==b
    whole.update(a);ph.update(b);regenerated.update(a);left-=len(a)
   assert not p.read(1)
  assert ph.hexdigest()==regenerated.hexdigest()==row['sha256']
  assert target.stat().st_mtime_ns==partst.st_mtime_ns and target.stat().st_size==partst.st_size
  rows.append(dict(path=str(target),name=row['name'],bytes=row['bytes'],sha256=row['sha256'],offset=offset,mtime_ns=partst.st_mtime_ns,
   github_asset_id=assets[row['name']]['id'],github_digest=assets[row['name']]['digest'],regenerated_range_sha256=regenerated.hexdigest()))
  offset+=row['bytes']
 assert not archive.read(1)
assert whole.hexdigest()==metadata['archive_sha256'] and offset==st.st_size
assert full.stat().st_mtime_ns==st.st_mtime_ns
# Verify every member's original bytes, without loading a model or parsing data.
expected={x['path']:x for x in manifest['files']};seen=set();member_bytes=0
with tarfile.open(full,'r|gz') as archive:
 for m in archive:
  if not m.isfile():continue
  n=m.name.removeprefix('./');assert n in expected and n not in seen
  x=expected[n];assert m.size==x['size'];h=hashlib.sha256()
  f=archive.extractfile(m)
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  assert h.hexdigest()==x['sha256'];seen.add(n);member_bytes+=m.size
assert seen==set(expected) and len(seen)==metadata['files']
proof=dict(schema='textop.user_authorized_storage_cleanup.v1',status='D_DUPLICATE_TRANSPORT_PARTS_RECOVERABILITY_VERIFIED',
 allowed_root=str(F),retained_archive=dict(path=str(full),bytes=st.st_size,sha256=whole.hexdigest(),mtime_ns=st.st_mtime_ns),
 retained_manifest=dict(path=str(F/metadata['manifest']),sha256=metadata['manifest_sha256']),
 targets=rows,total_duplicate_transport_bytes=sum(x['bytes'] for x in rows),
 all_original_archive_members_sha256_verified=len(seen),uncompressed_member_bytes=member_bytes,
 actual_range_regeneration_to_hash_sink=True,restore='Read each recorded offset/length from the retained full archive; verify per-part SHA256. Remote original uploaded parts also remain.',
 github_release=release['url'],github_all_target_asset_digests_verified=True,
 datasets_or_checkpoints_selected_for_deletion=False,full_archive_retained=True,original_results_retained=True,
 observed_unix=time.time(),verifier_sha256=sha(__file__))
(E/'D_recovery_verified.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
code=r'''
import hashlib,json,os,pathlib,shutil,stat,time
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
p=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/cache/downloads/IsaacGym_Preview_4.tar.gz')
allowed=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/cache/downloads')
assert p.resolve()==p and p.parent==allowed and allowed.stat().st_uid==1013
s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1013 and s.st_nlink==1 and s.st_size==201141673
for q in [p,allowed,allowed.parent,allowed.parent.parent]:assert q.stat().st_uid==1013 and not q.is_symlink()
finger=lambda st:dict(inode=st.st_ino,device=st.st_dev,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,owner_uid=st.st_uid)
before=finger(s);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert finger(p.stat())==before
refs=[];unreadable=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=1013:continue
  cmd=(proc/'cmdline').read_bytes()
  if str(p).encode() in cmd:refs.append(dict(pid=int(proc.name),kind='argv'))
  for fd in (proc/'fd').iterdir():
   try:
    if os.readlink(fd)==str(p):refs.append(dict(pid=int(proc.name),kind='fd'))
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:unreadable.append(int(proc.name))
assert not refs and not unreadable,(refs,unreadable)
sdk=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym');assert sdk.is_dir()
print(json.dumps(dict(status='IDLE_SDK_INSTALLER_SOURCE_VERIFIED',path=str(p),source_fingerprint=before,sha256=h.hexdigest(),
 active_references=refs,owned_processes_unreadable=unreadable,extracted_sdk_directory_retained=str(sdk),
 source_file_deleted=False,new_gpu_contexts=0,new_model_loads=0,observed_unix=time.time(),data_free_bytes=shutil.disk_usage(p).free)))
'''
server=json.loads(recorded_remote(code,E/'server_installer_verify_client.json',timeout=90))
(E/'server_installer_source_verified.json').write_text(json.dumps(server,indent=2),encoding='utf-8')
print(json.dumps(dict(D_duplicate_bytes=proof['total_duplicate_transport_bytes'],all_member_sha_verified=len(seen),server_software_bytes=server['source_fingerprint']['bytes'],files_deleted=0)))
