"""Restore an exact removed transport part from the retained complete archive."""
import argparse,hashlib,json,os,pathlib,shutil
from filter_recovery_windows_paths import guard_local
p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--part',required=True);p.add_argument('--output-directory',required=True);a=p.parse_args()
j=json.loads(pathlib.Path(a.proof).read_text(encoding='utf-8'));row=next(x for x in j['targets'] if x['name']==a.part)
root=pathlib.Path('D:/具身智能');src=pathlib.Path(j['retained_archive']['path']);out=pathlib.Path(a.output_directory).absolute()
guard_local(root,src);guard_local(root,out,directory=True)
assert out.is_dir() and shutil.disk_usage(out).free>=2*1024**3+row['bytes']+8*1024**2
final=out/row['name'];partial=out/(row['name']+'.restore.partial')
guard_local(root,final,must_absent=True);guard_local(root,partial,must_absent=True)
h=hashlib.sha256();left=row['bytes']
with src.open('rb') as f,partial.open('xb') as g:
 f.seek(row['offset'])
 while left:
  block=f.read(min(left,8*1024**2));assert block
  g.write(block);h.update(block);left-=len(block)
 g.flush();os.fsync(g.fileno())
assert h.hexdigest()==row['sha256'];partial.rename(final)
print(json.dumps(dict(path=str(final),bytes=final.stat().st_size,sha256=h.hexdigest())))
