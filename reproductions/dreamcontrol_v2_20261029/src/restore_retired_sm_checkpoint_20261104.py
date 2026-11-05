"""Restore one hash-verified retired checkpoint, retaining its compressed archive."""
import argparse,gzip,hashlib,json,os,pathlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INDEX=R/'checkpoint_archival_20261104/retired_sm_v2_manifest.json'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',choices=['resume_early_96.pt','resume_failure_3424.pt'],required=True);a=p.parse_args()
m=json.loads(INDEX.read_text());assert m['status']=='retired_sm_snapshots_losslessly_cold_archived'
r=next(x for x in m['records'] if pathlib.Path(x['source']).name==a.name);target=pathlib.Path(r['source']);archive=pathlib.Path(r['archive'])
assert target.parent.resolve()==(R/'stage3_supermimic29_20261103/runs/pick500_8912_72000_parallel_v2').resolve()
assert sha(archive)==r['archive_sha256']
if target.exists():
 assert target.stat().st_size==r['source_bytes'] and sha(target)==r['source_sha256'];print('Existing original already verified; no overwrite.')
else:
 assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2+r['source_bytes']+32*1024**2
 temporary=target.with_suffix('.restore.tmp');hasher=hashlib.sha256();size=0
 with gzip.open(archive,'rb') as src,temporary.open('xb') as dst:
  for chunk in iter(lambda:src.read(4*1024**2),b''):dst.write(chunk);hasher.update(chunk);size+=len(chunk)
  dst.flush();os.fsync(dst.fileno())
 assert size==r['source_bytes'] and hasher.hexdigest()==r['source_sha256'] and sha(temporary)==r['source_sha256']
 # Same filesystem hard-link is exclusive: refuse a concurrently created target.
 os.link(temporary,target);temporary.unlink()
 receipt={'status':'checkpoint_restored_verified_archive_retained','source':str(target),'sha256':sha(target),'archive':str(archive),'archive_sha256':sha(archive)}
 pathlib.Path(str(target)+'.restored.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
