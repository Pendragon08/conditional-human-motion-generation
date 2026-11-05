"""Losslessly compress only closed logs of fully completed production cohorts."""
import gzip,hashlib,json,os,pathlib,shutil,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');W=R/'stage3_supported_references_20261104/bulk_v1';FLOOR=8*1024**3+456*1024**2
status=W/'log_archive_status.json';assert not status.exists();state={'status':'watching_completed_cohorts','pid':os.getpid(),'archived_files':0,'saved_bytes':0}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(status)
def opened(p):
 for process in pathlib.Path('/proc').iterdir():
  if not process.name.isdigit():continue
  try:
   if process.stat().st_uid!=os.getuid():continue
   for fd in (process/'fd').iterdir():
    try:
     if fd.resolve()==p:return True
    except (FileNotFoundError,PermissionError):pass
  except (FileNotFoundError,PermissionError):pass
 return False
try:
 save()
 while True:
  for summary in sorted((W/'cohorts').glob('*/summary.json')):
   try:j=json.loads(summary.read_text())
   except json.JSONDecodeError:continue
   if j.get('status')!='cohort_completed':continue
   for original in sorted(summary.parent.glob('*.log')):
    p=original.resolve();assert W.resolve() in p.parents and not original.is_symlink() and p.stat().st_uid==os.getuid()
    st=p.stat()
    if st.st_size<256*1024 or st.st_size>256*1024**2 or opened(p):continue
    target=p.with_name(p.name+'.gz');temp=p.with_name(p.name+'.gz.partial');marker=p.with_name(p.name+'.archived.json')
    if target.exists() or temp.exists() or marker.exists():continue
    if shutil.disk_usage('/data').free<FLOOR+20*1024**2:continue
    before=sha(p)
    with p.open('rb') as source,temp.open('xb') as output:
     with gzip.GzipFile(fileobj=output,mode='wb',compresslevel=3,mtime=0,filename='') as compressed:
      for chunk in iter(lambda:source.read(1024**2),b''):compressed.write(chunk)
     output.flush();os.fsync(output.fileno())
    verified=hashlib.sha256()
    with gzip.open(temp,'rb') as f:
     for chunk in iter(lambda:f.read(1024**2),b''):verified.update(chunk)
    current=p.stat();assert (st.st_ino,st.st_dev,st.st_size)==(current.st_ino,current.st_dev,current.st_size) and verified.hexdigest()==before==sha(p)
    if opened(p):temp.unlink();continue
    compressed_sha=sha(temp);temp.rename(target)
    entry={'original':str(p),'archive':str(target),'original_bytes':st.st_size,'original_sha256':before,'gzip_bytes':target.stat().st_size,'gzip_sha256':compressed_sha,'complete_decompression_verified':True,'source_removed':False,'restore':'decompress gzip to exact original path and verify original_sha256; preserve archive'}
    marker.open('x').write(json.dumps(entry,indent=2));p.unlink();entry['source_removed']=True;marker.write_text(json.dumps(entry,indent=2))
    with (W/'completed_log_archive_manifest.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    state['archived_files']+=1;state['saved_bytes']+=st.st_size-entry['gzip_bytes'];save()
  control=json.loads((W/'controller_status.json').read_text())
  if control['status'] in ['failed_preserved','all1500_scene_references_admitted_final_mixed_loader_pending']:
   state.update(status='completed_log_archival_finished',controller_status=control['status']);save();break
  time.sleep(20)
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save();raise
