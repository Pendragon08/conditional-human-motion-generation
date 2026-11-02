"""Archive idle packages/caches before deletion; relocate assets without loss.

Fixed personal roots only. No user dataset, experiment checkpoint or unique
record is a deletion target. Current TextOp/GMR environments are excluded.
"""
import argparse,datetime,gzip,hashlib,json,os,pathlib,shutil,stat,subprocess,tarfile,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh')
TASK=ROOT/'textop-reproduction-20261024'
WORK=pathlib.Path('/dev/shm/ouyanghao_storage_20261025')
ENV=ROOT/'humanoid-combat-motion/.venv-engineai-official'
HOME=pathlib.Path('/home/ouyanghao')
CACHES=[ROOT/'humanoid-combat-motion/cache'/x for x in [
 'r3_viewer_kit/cache/nv_shadercache','r3_viewer_kit/cache/DerivedDataCache',
 'pycache','pycache-timotion','r3_pip','pip','pip-pbhc-training','xdg/pip','engineai-official/wheels']]
CACHES +=[ROOT/'phygile-reproduction-20261017/cache/xdg/ov/texturecache',
 HOME/'.cache/pip',HOME/'.nv/ComputeCache',HOME/'paper-faithful-compile-cache-20261022']
SOURCES=[ENV]+CACHES
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(name,j):
 p=TASK/'receipts'/('storage_'+name+'_20261025.json');p.write_text(json.dumps(j,indent=2));return p
def assert_personal(p):
 assert os.getuid()==1013 and p.is_absolute()
 assert p==ROOT or ROOT in p.parents or p==HOME or HOME in p.parents
 assert p.lstat().st_uid==1013
def idle(prefixes):
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=1013:continue
   command=(p/'cmdline').read_bytes().decode(errors='replace');maps=(p/'maps').read_text(errors='replace')
   assert not any(str(x)+'/' in command or str(x)+'/' in maps for x in prefixes),'Selected root used by process'+p.name
   for fd in (p/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert not any(target.startswith(str(x)+'/') for x in prefixes),'Selected root has open file'+p.name
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
def walk(root):
 assert_personal(root);assert not root.is_symlink()
 for d,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   p=pathlib.Path(d,name);assert_personal(p);s=p.lstat()
   assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode)
   yield p,s
  dirs[:]=[x for x in dirs if not pathlib.Path(d,x).is_symlink()]
def archive():
 assert ENV.exists() and not WORK.exists()
 idle(SOURCES);WORK.mkdir(mode=0o700)
 cfg=(ENV/'pyvenv.cfg').read_text();assert 'world-model-risk-planning/.env/bin' in cfg
 # Refuse deleting a base interpreter or linked dependency of any other env.
 dependencies=[]
 for d,dirs,files in os.walk(ROOT,followlinks=False):
  dirs[:]=[x for x in dirs if not pathlib.Path(d,x).is_symlink()]
  if pathlib.Path(d)==ENV or ENV in pathlib.Path(d).parents:continue
  if 'pyvenv.cfg' in files and str(ENV) in pathlib.Path(d,'pyvenv.cfg').read_text(errors='replace'):dependencies.append(d)
  for n in files:
   p=pathlib.Path(d,n)
   if p.is_symlink():
    try:r=p.resolve()
    except RuntimeError:continue
    if r==ENV or ENV in r.parents:dependencies.append(str(p))
 assert not dependencies,dependencies
 lock=subprocess.check_output([str(ENV/'bin/python'),'-m','pip','freeze','--all'],text=True)
 (WORK/'requirements-engineai.txt').write_text(lock);(WORK/'pyvenv.cfg').write_text(cfg)
 rows=[];archive=WORK/'inactive_engineai_and_caches.tar.gz'
 proof={'status':'ARCHIVING','time_beijing':now(),'source_code_sha256':sha(__file__),'source_roots':[str(x) for x in SOURCES],
        'environment_idle':True,'external_environment_dependencies':dependencies,'files_archived':0,'source_bytes':0,
        'datasets_deleted':0,'experiment_checkpoints_deleted':0,'unique_records_deleted':0}
 save('archive_progress',proof)
 with tarfile.open(archive,'w:gz',compresslevel=1,dereference=False) as tar:
  for root in SOURCES:
   if not root.exists():continue
   all_rows=[(root,root.lstat())]+list(walk(root))
   for path,s in all_rows:
    before=path.lstat();name=path.as_posix().lstrip('/');info=tar.gettarinfo(str(path),arcname=name)
    row={'path':str(path),'member':name,'mode':stat.S_IMODE(s.st_mode),'bytes':s.st_size,'allocated':s.st_blocks*512,
         'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'type':'directory' if stat.S_ISDIR(s.st_mode) else 'symlink' if stat.S_ISLNK(s.st_mode) else 'file'}
    if stat.S_ISREG(s.st_mode):
     digest=hashlib.sha256()
     class Reader:
      def __init__(self,f):self.f=f
      def read(self,n=-1):
       b=self.f.read(n);digest.update(b);return b
     with path.open('rb') as f:tar.addfile(info,Reader(f))
     row['sha256']=digest.hexdigest();proof['files_archived']+=1;proof['source_bytes']+=s.st_size
    else:
     tar.addfile(info)
     if stat.S_ISLNK(s.st_mode):row['link_target']=os.readlink(path)
    after=path.lstat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    rows.append(row)
    if len(rows)%5000==0:save('archive_progress',proof)
 manifest=WORK/'source_manifest.json';manifest.write_text(json.dumps(rows));lookup={x['member']:x for x in rows}
 checked=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   if not member.isfile():continue
   row=lookup[member.name];h=hashlib.sha256()
   with tar.extractfile(member) as f:
    for b in iter(lambda:f.read(8388608),b''):h.update(b)
   assert h.hexdigest()==row['sha256'];checked+=1
 assert checked==proof['files_archived']
 pieces=[];full=hashlib.sha256()
 with archive.open('rb') as f:
  index=1
  while True:
   block=f.read(8*1024**2)
   if not block:break
   part=WORK/('inactive_engineai_and_caches.tar.gz.part%03d'%index);count=0;h=hashlib.sha256()
   with part.open('xb') as target:
    while block:
     target.write(block);h.update(block);full.update(block);count+=len(block)
     if count==1024**3:break
     block=f.read(min(8388608,1024**3-count))
   pieces.append({'name':part.name,'bytes':count,'sha256':h.hexdigest()});index+=1
 proof.update(status='ARCHIVE_CRC_AND_EVERY_FILE_SHA_VERIFIED',verified_files=checked,archive_bytes=archive.stat().st_size,
              archive_sha256=full.hexdigest(),manifest_sha256=sha(manifest),parts=pieces,completed_beijing=now())
 (WORK/'backup_index.json').write_text(json.dumps(proof,indent=2));save('archive',proof);print(json.dumps(proof),flush=True)
def apply():
 proof=json.loads((WORK/'backup_index.json').read_text());auth=json.loads((WORK/'github_backup_verified.json').read_text())
 assert auth['all_asset_digests_verified'] and auth['code_and_inventory_backed_up_commit'] and proof['source_code_sha256']==sha(__file__)
 assert auth['archive_sha256']==proof['archive_sha256'] and auth['manifest_sha256']==proof['manifest_sha256']
 assert [{k:v for k,v in x.items() if k in ['name','bytes','sha256']} for x in auth['parts']]==proof['parts']
 rows=json.loads((WORK/'source_manifest.json').read_text());idle(SOURCES)
 for row in rows:
  p=pathlib.Path(row['path']);assert_personal(p);s=p.lstat()
  assert any(p==x or x in p.parents for x in SOURCES)
  assert s.st_ino==row['inode'] and s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns']
  if row['type']=='file':assert sha(p)==row['sha256']
  elif row['type']=='symlink':assert os.readlink(p)==row['link_target']
 before={x:shutil.disk_usage(x).free for x in ['/data','/home/ouyanghao']};removed=0
 for row in rows:
  if row['type']=='directory':continue
  p=pathlib.Path(row['path']);assert_personal(p);p.unlink();removed+=1
 for row in sorted(rows,key=lambda x:len(pathlib.Path(x['path']).parts),reverse=True):
  if row['type']=='directory':
   p=pathlib.Path(row['path']);assert_personal(p);p.rmdir()
 receipt={'status':'BACKED_UP_IDLE_ENVIRONMENT_AND_CACHES_REMOVED','time_beijing':now(),'removed_files_and_links':removed,
          'free_before':before,'free_after':{x:shutil.disk_usage(x).free for x in before},'github_backup':auth,
          'datasets_deleted':0,'experiment_checkpoints_deleted':0,'unique_records_deleted':0,'training_process_stopped':False,
          'archived_environment':str(ENV),'current_textop_and_gmr_environments_preserved':True}
 save('cleanup',receipt)
 ENV.with_name(ENV.name+'.ARCHIVED_20261025.json').write_text(json.dumps({'restore_release':auth['release_url'],'archive_sha256':proof['archive_sha256'],
  'restore':'download ordered release pieces, concatenate, then tar -xzf archive -C / ; original base interpreter and editable source trees remain'},indent=2))
 print(json.dumps(receipt),flush=True)
def relocate():
 assert json.loads((TASK/'receipts/storage_cleanup_20261025.json').read_text())['status']=='BACKED_UP_IDLE_ENVIRONMENT_AND_CACHES_REMOVED'
 target=ROOT/'retained-system-assets-20261025';assert not target.exists();target.mkdir(mode=0o700)
 result={'status':'RELOCATING_WITH_SHA_CHECK','time_beijing':now(),'moves':[],
         'free_before':{x:shutil.disk_usage(x).free for x in ['/data','/home/ouyanghao']},'datasets_deleted':0,'checkpoints_deleted':0}
 for source in [HOME/'textop-assets-20261024',HOME/'phygile-preserved-assets-20261023']:
  idle([source]);rows=list(walk(source));assert all(not stat.S_ISLNK(s.st_mode) for p,s in rows)
  total=sum(s.st_size for p,s in rows if stat.S_ISREG(s.st_mode));assert shutil.disk_usage(target).free>total+2*1024**3
  dest=target/source.name;dest.mkdir(mode=stat.S_IMODE(source.stat().st_mode));checked=[]
  for p,s in rows:
   rel=p.relative_to(source);q=dest/rel
   if stat.S_ISDIR(s.st_mode):q.mkdir(mode=stat.S_IMODE(s.st_mode),exist_ok=True);continue
   digest=sha(p);shutil.copy2(p,q)
   with q.open('rb') as f:os.fsync(f.fileno())
   assert sha(q)==digest;checked.append({'relative_path':rel.as_posix(),'bytes':s.st_size,'sha256':digest})
  move={'source':str(source),'destination':str(dest),'files_verified':len(checked),'bytes':total,'files':checked,'phase':'COPY_SHA_VERIFIED'}
  result['moves'].append(move);save('relocation',result)
  old=source.with_name(source.name+'.relocation_old_20261025');link=source.with_name(source.name+'.new_link_20261025')
  assert not old.exists() and not link.exists();link.symlink_to(dest,target_is_directory=True)
  source.rename(old);os.replace(link,source);assert source.resolve()==dest
  for row in checked:assert sha(source/row['relative_path'])==row['sha256']
  move['phase']='ORIGINAL_PATH_LINK_VERIFIED';save('relocation',result)
  for p,s in list(walk(old)):
   if stat.S_ISREG(s.st_mode):assert_personal(p);p.unlink()
  for p,s in sorted(list(walk(old)),key=lambda x:len(x[0].parts),reverse=True):
   if stat.S_ISDIR(s.st_mode):p.rmdir()
  old.rmdir();move['phase']='RELOCATED_ORIGINAL_PATH_PRESERVED';save('relocation',result)
 result.update(status='ALL_ASSETS_RELOCATED_SHA_VERIFIED_ORIGINAL_PATHS_PRESERVED',
               free_after={x:shutil.disk_usage(x).free for x in ['/data','/home/ouyanghao']},completed_beijing=now())
 save('relocation',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['archive','apply','relocate']);a=p.parse_args()
 assert_personal(ROOT);assert ROOT.resolve()==ROOT and HOME.resolve()==HOME
 {'archive':archive,'apply':apply,'relocate':relocate}[a.phase]()
