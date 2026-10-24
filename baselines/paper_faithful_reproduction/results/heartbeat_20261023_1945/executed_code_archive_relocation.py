"""One-time server relocation of an immutable archive; preserve its original path.

Copies and fsyncs the exact bytes, verifies SHA256 and unchanged source stat,
then atomically replaces the original directory entry with a verified symlink.
The full dataset archive remains on the server. No cleanup of any dataset,
checkpoint, result, code, or rollback artifact is performed.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,datetime

SOURCE=Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/storage_cleanup_code_packages_20261023_v2/inactive_source_packages_20261023_v2.tar.gz')
DEST=Path('/home/ouyanghao/phygile-preserved-assets-20261023/inactive_source_packages_20261023_v2.tar.gz')
RECEIPT=Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/storage_cleanup_code_packages_20261023_v2/code_archive_relocation_receipt.json')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if SOURCE.is_symlink() or DEST.exists() or RECEIPT.exists():raise RuntimeError('Already relocated or existing destination; do not overwrite')
 assert SOURCE.resolve().is_relative_to(Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/storage_cleanup_code_packages_20261023_v2'))
 assert DEST.parent==Path('/home/ouyanghao/phygile-preserved-assets-20261023')
 used=subprocess.run(['fuser',str(SOURCE)],capture_output=True,text=True)
 if used.returncode!=1:raise RuntimeError('Archive is open or fuser inspection failed; preserve it in place')
 stat=SOURCE.stat()
 assert os.getuid()==1013 and stat.st_uid==1013 and DEST.parent.stat().st_uid==1013
 assert digest(SOURCE)=='6b67c9e3d3b7528d475cd648d0b8fb35ed16f832db3eee1dd433965bc5593141'
 assert stat.st_size==182701714
 assert shutil.disk_usage(DEST.parent.parent).free>stat.st_size+1024**3
 before={'data_free':shutil.disk_usage(SOURCE).free,'home_free':shutil.disk_usage(DEST.parent.parent).free}
 DEST.parent.mkdir(exist_ok=True)
 with SOURCE.open('rb') as src,DEST.open('xb') as dst:
  shutil.copyfileobj(src,dst,length=8*1024*1024);dst.flush();os.fsync(dst.fileno())
 shutil.copystat(SOURCE,DEST)
 original=digest(SOURCE);copied=digest(DEST)
 assert original==copied and SOURCE.stat().st_size==stat.st_size and SOURCE.stat().st_mtime_ns==stat.st_mtime_ns
 assert subprocess.run(['fuser',str(SOURCE)],capture_output=True).returncode==1
 link=SOURCE.with_name(SOURCE.name+'.preserved-link')
 os.symlink(str(DEST),link)
 assert link.resolve()==DEST and digest(link)==original
 os.replace(link,SOURCE)
 assert SOURCE.is_symlink() and SOURCE.resolve()==DEST and digest(SOURCE)==original
 record={'status':'PRESERVED_AND_RELOCATED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'original_access_path':str(SOURCE),'physical_archive_path':str(DEST),'bytes':stat.st_size,'sha256':original,
  'source_copy_and_link_sha256_equal':True,'original_path_readable':True,'archive_contents_preserved':True,
  'datasets_deleted':False,'checkpoints_touched':False,'training_processes_touched':False,
  'space_before':before,'space_after':{'data_free':shutil.disk_usage(SOURCE.parent).free,'home_free':shutil.disk_usage(DEST).free}}
 RECEIPT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
