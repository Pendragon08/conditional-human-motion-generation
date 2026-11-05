"""Delete only the audited project-owned derived NPZ files explicitly requested.

The immutable manifest defines exact paths, contents, sizes and timestamps.
Raw GRAB archives are independently rehashed against the recorded registry
before unlinking. No recursion, source archive, model or OmniRetarget deletion.
"""
import hashlib,json,os,stat,subprocess
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029').resolve(strict=True)
PLAN=ROOT/'gmr_reset_cleanup_plan_20261103.json'
PLAN_SHA='4f7149a47be657d28840b12a369d09e4339932a1635dd881909c54f2a5f38602'
REG=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
REG_SHA='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def valid(row):
 relative=Path(row['path'])
 if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.npz':raise ValueError('unsafe path')
 p=ROOT/relative
 if p.is_symlink() or not p.resolve(strict=True).is_relative_to(ROOT) or any(x.is_symlink() for x in p.parents if x!=ROOT and x.is_relative_to(ROOT)):raise ValueError('unsafe file path')
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size!=row['bytes'] or s.st_mtime_ns!=row['mtime_ns'] or sha(p)!=row['sha256']:raise ValueError('file changed: '+row['path'])
 return p
def main():
 if sha(PLAN)!=PLAN_SHA or sha(REG)!=REG_SHA:raise ValueError('plan or registry changed')
 plan=json.loads(PLAN.read_text());rows=plan['files']
 if len(rows)!=7099 or len({r['path'] for r in rows})!=len(rows) or plan['missing_dirs']:raise ValueError('plan coverage')
 processes=subprocess.check_output(['ps','-eo','args='],text=True)
 for token in ('retarget_grab_full_source_batch_v2_20261103.py','reconstruct_grab_full_source_batch_v2_20261102.py','bmlmovi_stream_consumer_20261102.py'):
  if token in processes:raise RuntimeError('active writer: '+token)
 registry=[json.loads(x) for x in REG.read_text().splitlines()];expected={}
 for row in registry:
  name=row['archive'];value=row['archive_sha256']
  if name in expected and expected[name]!=value:raise ValueError('inconsistent raw archive hash')
  expected[name]=value
 for i in range(1,11):
  name=f'grab__s{i}.zip';path=ROOT/'data/GRAB'/name
  if not path.is_file() or path.is_symlink() or sha(path)!=expected.get(name):raise ValueError('raw GRAB archive missing or changed: '+name)
 for row in rows:valid(row)
 if sha(PLAN)!=PLAN_SHA or sha(REG)!=REG_SHA:raise RuntimeError('inputs changed')
 journal=ROOT/'gmr_reset_cleanup_journal_20261103.jsonl'
 with journal.open('x',encoding='utf8',buffering=1) as out:
  for i,row in enumerate(rows):
   p=valid(row);p.unlink()
   out.write(json.dumps({'event':'deleted_derived_npz','path':row['path'],'bytes':row['bytes'],'sha256':row['sha256']})+'\n')
   if i%100==0:out.flush();os.fsync(out.fileno())
  out.flush();os.fsync(out.fileno())
 result={'deleted':len(rows),'bytes':sum(x['bytes'] for x in rows),'plan_sha256':PLAN_SHA,'journal_sha256':sha(journal),'raw_grab_archives_verified':10,'training_ready':False,'scope':'Only plan-listed project derived NPZ files. Other outputs require additional classification.'}
 with (ROOT/'gmr_reset_cleanup_result_20261103.json').open('x',encoding='utf8') as out:json.dump(result,out,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
