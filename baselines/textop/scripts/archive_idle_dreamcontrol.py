"""Recover space from one idle software overlay only, after verified backup.

No interpreter, dataset, checkpoint, results directory or active package tree
is a cleanup target. PBHC training is explicitly excluded: TextOp depends on it.
"""
import argparse,datetime,hashlib,json,os,pathlib,shutil,stat,tarfile

ROOT=pathlib.Path('/data/code/ouyanghao/oyh')
SOURCE=ROOT/'humanoid-combat-motion/envs/dreamcontrol-retarget'
TASK=ROOT/'textop-reproduction-20261024'
WORK=pathlib.Path('/dev/shm/ouyanghao_dreamcontrol_backup_20261026')
RECEIPTS=TASK/'receipts/idle_dreamcontrol_20261026'
REPO='Pendragon08/conditional-human-motion-generation'
TAG='inactive-dreamcontrol-overlay-20261026'

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def save(name,data):
 RECEIPTS.mkdir(exist_ok=True)
 data['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=RECEIPTS/(name+'.json');q=p.with_suffix('.partial')
 q.write_text(json.dumps(data,indent=2));q.replace(p)

def safe(p):
 assert os.getuid()==1013 and (p==SOURCE or SOURCE in p.parents)
 assert not SOURCE.is_symlink() and SOURCE.resolve()==SOURCE
 assert p.lstat().st_uid==1013

def rows_now():
 safe(SOURCE)
 rows=[(SOURCE,SOURCE.lstat())]
 for d,dirs,files in os.walk(SOURCE,followlinks=False):
  for name in dirs+files:
   p=pathlib.Path(d,name);safe(p);s=p.lstat()
   assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode)
   rows.append((p,s))
  dirs[:]=[n for n in dirs if not pathlib.Path(d,n).is_symlink()]
 return rows

def idle_and_independent():
 references=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if p.stat().st_uid!=1013:continue
   for name in ['cmdline','maps','environ']:
    if str(SOURCE).encode() in (p/name).read_bytes():references.append([p.name,name])
   for fd in (p/'fd').iterdir():
    try:
     if os.readlink(fd).startswith(str(SOURCE)+'/'):references.append([p.name,'fd'])
    except OSError:pass
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not references,'Selected software is in use: '+repr(references)
 dependencies=[]
 for d,dirs,files in os.walk(ROOT,followlinks=False):
  current=pathlib.Path(d)
  if current==SOURCE or SOURCE in current.parents:dirs[:]=[];continue
  for n in dirs+files:
   p=pathlib.Path(d,n)
   if p.is_symlink():
    try:
     resolved=p.resolve()
     if resolved==SOURCE or SOURCE in resolved.parents:dependencies.append(str(p))
    except (OSError,RuntimeError):pass
  for n in files:
   if n=='pyvenv.cfg' or n.endswith('.pth'):
    p=pathlib.Path(d,n)
    try:
     if p.stat().st_size<=1024**2 and str(SOURCE) in p.read_text(errors='replace'):dependencies.append(str(p))
    except OSError:pass
  dirs[:]=[n for n in dirs if not pathlib.Path(d,n).is_symlink()]
 assert not dependencies,'Another runtime depends on this overlay: '+repr(dependencies)
 return {'active_references':references,'runtime_dependencies':dependencies}

def archive():
 assert not WORK.exists() and not (RECEIPTS/'archive.json').exists()
 assert shutil.disk_usage('/dev/shm').free>44*1024**3,'Preserve capacity for GMR/KIT working data'
 checks=idle_and_independent();source_rows=rows_now();WORK.mkdir(mode=0o700)
 # Record versions without importing this idle package environment.
 packages=[]
 for p in sorted(SOURCE.glob('*.dist-info/METADATA')):
  fields={}
  for line in p.read_text(errors='replace').splitlines():
   if line.startswith(('Name: ','Version: ')):
    k,v=line.split(': ',1);fields[k]=v
   if len(fields)==2:break
  packages.append(fields)
 (WORK/'packages.json').write_text(json.dumps(packages,indent=2))
 state={'status':'ARCHIVING_IDLE_SOFTWARE_ONLY','source':str(SOURCE),'source_script_sha256':sha(__file__),
        'source_bytes':sum(s.st_size for p,s in source_rows if stat.S_ISREG(s.st_mode)),
        'files_archived':0,'datasets_deleted':0,'checkpoints_deleted':0,'records_deleted':0,**checks}
 save('progress',state);rows=[];archive_path=WORK/'idle_dreamcontrol_overlay.tar.gz'
 with tarfile.open(archive_path,'w:gz',compresslevel=1,dereference=False) as tar:
  for p,s in source_rows:
   safe(p);member=p.as_posix().lstrip('/');info=tar.gettarinfo(str(p),arcname=member)
   row={'path':str(p),'member':member,'mode':stat.S_IMODE(s.st_mode),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
        'inode':s.st_ino,'device':s.st_dev,'type':'directory' if stat.S_ISDIR(s.st_mode) else 'symlink' if stat.S_ISLNK(s.st_mode) else 'file'}
   if stat.S_ISREG(s.st_mode):
    # Store each regular file independently, including hard-linked source files.
    info.type=tarfile.REGTYPE;info.linkname='';info.size=s.st_size;h=hashlib.sha256()
    class Reader:
     def __init__(self,f):self.f=f
     def read(self,n=-1):
      value=self.f.read(n);h.update(value);return value
    with p.open('rb') as f:tar.addfile(info,Reader(f))
    row['sha256']=h.hexdigest();state['files_archived']+=1
   else:
    tar.addfile(info)
    if stat.S_ISLNK(s.st_mode):row['link_target']=os.readlink(p)
   after=p.lstat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
   rows.append(row)
   if len(rows)%2000==0:save('progress',state)
 manifest=WORK/'source_manifest.json';manifest.write_text(json.dumps(rows))
 lookup={r['member']:r for r in rows};checked=0;seen=set()
 with tarfile.open(archive_path,'r:gz') as tar:
  for member in tar:
   assert member.name not in seen;seen.add(member.name);row=lookup[member.name]
   assert member.mode==row['mode']
   if row['type']=='file':
    assert member.isfile() and member.size==row['bytes'];h=hashlib.sha256()
    with tar.extractfile(member) as f:
     for block in iter(lambda:f.read(8388608),b''):h.update(block)
    assert h.hexdigest()==row['sha256'];checked+=1
   elif row['type']=='symlink':assert member.issym() and member.linkname==row['link_target']
   else:assert member.isdir()
 assert len(seen)==len(rows) and checked==state['files_archived']
 parts=[];full=hashlib.sha256()
 with archive_path.open('rb') as f:
  index=1
  while True:
   block=f.read(8388608)
   if not block:break
   p=WORK/('idle_dreamcontrol_overlay.tar.gz.part%03d'%index);count=0;h=hashlib.sha256()
   with p.open('xb') as target:
    while block:
     target.write(block);h.update(block);full.update(block);count+=len(block)
     if count==1024**3:break
     block=f.read(min(8388608,1024**3-count))
   parts.append({'name':p.name,'bytes':count,'sha256':h.hexdigest()});index+=1
 state.update(status='ARCHIVE_ALL_MEMBERS_AND_FILE_HASHES_VERIFIED',verified_files=checked,
              archive_bytes=archive_path.stat().st_size,archive_sha256=full.hexdigest(),manifest_sha256=sha(manifest),
              packages_sha256=sha(WORK/'packages.json'),parts=parts,source_deleted=False)
 (WORK/'archive_index.json').write_text(json.dumps(state,indent=2));save('archive',state);print(json.dumps(state),flush=True)

def apply():
 proof=json.loads((WORK/'archive_index.json').read_text());auth=json.loads((WORK/'github_verified.json').read_text())
 assert proof['status']=='ARCHIVE_ALL_MEMBERS_AND_FILE_HASHES_VERIFIED' and proof['source_script_sha256']==sha(__file__)
 assert auth['repository']==REPO and auth['tag']==TAG and auth['all_digests_verified'] and auth['code_inventory_commit']
 assert auth['archive_sha256']==proof['archive_sha256'] and auth['manifest_sha256']==proof['manifest_sha256']
 assert auth['parts']==proof['parts'] and sha(WORK/'source_manifest.json')==proof['manifest_sha256']
 idle_and_independent();rows=json.loads((WORK/'source_manifest.json').read_text())
 assert {str(p) for p,s in rows_now()}=={r['path'] for r in rows},'Source inventory changed'
 for r in rows:
  p=pathlib.Path(r['path']);safe(p);s=p.lstat()
  assert (s.st_ino,s.st_size,s.st_mtime_ns)==(r['inode'],r['bytes'],r['mtime_ns'])
  if r['type']=='file':assert sha(p)==r['sha256']
  elif r['type']=='symlink':assert os.readlink(p)==r['link_target']
 idle_and_independent();before=shutil.disk_usage('/data').free
 save('apply_started',{'status':'BACKUP_AND_SOURCE_REVERIFIED','github':auth,'free_before':before,'source':str(SOURCE)})
 for r in rows:
  if r['type']!='directory':p=pathlib.Path(r['path']);safe(p);p.unlink()
 for r in sorted(rows,key=lambda r:len(pathlib.Path(r['path']).parts),reverse=True):
  if r['type']=='directory':p=pathlib.Path(r['path']);safe(p);p.rmdir()
 recovery={'release_url':auth['release_url'],'archive_sha256':proof['archive_sha256'],
           'restore':'Concatenate verified ordered parts, then tar -xzf archive -C /. Original path must be absent; never overwrite a new runtime.',
           'requires_validation_after_restore':True}
 SOURCE.with_name(SOURCE.name+'.ARCHIVED_20261026.json').write_text(json.dumps(recovery,indent=2))
 result={'status':'IDLE_DREAMCONTROL_SOFTWARE_BACKED_UP_AND_REMOVED','free_before':before,'free_after':shutil.disk_usage('/data').free,
         'source':str(SOURCE),'github':auth,'datasets_deleted':0,'checkpoints_deleted':0,'records_deleted':0,
         'active_textop_gmr_isaac_packages_preserved':True,'pbhc_training_base_preserved':True}
 save('cleanup',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['archive','apply']);args=parser.parse_args()
 assert ROOT.resolve()==ROOT and os.getuid()==1013
 os.sched_setaffinity(0,{0});os.nice(19)
 {'archive':archive,'apply':apply}[args.phase]()
