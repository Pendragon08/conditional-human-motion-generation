"""Restore one exact archived Pour intermediate; never overwrite different content."""
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path
from github_archive_stream_20261105_v2 import inspect_archive
from verify_pour_github_archives_20261105_v2 import R,OUT,ASSETS,digest,fixed_manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-path',required=True);args=ap.parse_args()
 token=json.loads(sys.stdin.buffer.read())['token'];plan=json.loads((OUT/'candidates_manifest.json').read_text());matches=[r for r in plan['rows'] if r['source_path']==args.source_path]
 if len(matches)!=1:raise RuntimeError('Only an exact enumerated source path may be restored')
 row=matches[0];destination=Path(row['source_path'])
 if destination.is_symlink() or destination.parent.resolve()!=R/'stage3_supported_references_20261104/task_ppo_20261104/pour/ppo_8912_5000_v1/ppo':raise RuntimeError('Noncanonical destination')
 if destination.exists():
  if destination.stat().st_size==row['bytes'] and digest(destination)==row['sha256']:
   print(json.dumps({'status':'already_restored_same_content','path':str(destination)}));return
  raise RuntimeError('Refusing to overwrite different existing content')
 disk=os.statvfs(destination.parent)
 if disk.f_bavail*disk.f_frsize-row['bytes']<11*1024**3:raise RuntimeError('Restoration would cross retained 11GiB free-space guard')
 asset=next(a for a in ASSETS if a['id']==row['asset_id']);manifest=fixed_manifest(asset)
 fd,temp_name=tempfile.mkstemp(prefix=destination.name+'.github-restore-',suffix='.pending',dir=destination.parent);temp=Path(temp_name)
 try:
  with os.fdopen(fd,'wb') as sink:
   result=inspect_archive(token,asset,manifest,restore_member=row['member'],restore_sink=sink);sink.flush();os.fsync(sink.fileno())
  if temp.stat().st_size!=row['bytes'] or digest(temp)!=row['sha256']:raise RuntimeError('Restored file digest mismatch; temporary retained')
  os.link(temp,destination);os.unlink(temp)
  out=OUT/('restore_'+destination.name+'.json');out.write_text(json.dumps({'status':'restored_verified_without_overwriting','path':str(destination),'bytes':row['bytes'],'sha256':row['sha256'],'full_archive_sha256':result['archive_sha256'],'archive_id':asset['id']},indent=2)+'\n')
  print(out.read_text())
 except BaseException:
  print(json.dumps({'status':'restore_failed_temporary_preserved','temporary_path':str(temp)}),file=sys.stderr);raise
 finally:token=''
if __name__=='__main__':main()
