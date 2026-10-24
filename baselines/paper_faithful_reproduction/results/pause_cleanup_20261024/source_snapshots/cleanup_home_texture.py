"""Remove only recovered, byte-identical built-in Wireframe texture caches."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,shutil,stat
from cleanup_verified_old_checkpoints import active_check

ROOT=Path('/home/ouyanghao/.cache/ov/texturecache')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def check(r):
 p=Path(r['path']);assert p.parent==ROOT and re.fullmatch(r'Wireframe[^/]*\.ovtex\.\d+',p.name)
 assert os.getuid()==1013 and p.resolve()==p
 for q in [p,*p.parents]:assert not q.is_symlink()
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1013 and s.st_nlink==1
 assert (s.st_ino,s.st_mtime_ns,s.st_size)==(r['inode'],r['mtime_ns'],r['bytes'])
 assert digest(p)==r['sha256'] and r['local_member_recovered_sha256']==r['sha256']
 return p,s

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
 assert not a.output.exists();plan=json.loads(a.plan.read_text());rows=plan['files']
 assert len(rows)==17 and plan['backup_archive_currently_verified']
 backup=plan['backup_verification'];assert backup['asset']['digest']=='sha256:'+backup['local_archive_sha256']
 assert backup['asset']['url'].startswith('https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/')
 paths=[];allocated=0
 for r in rows:
  p,s=check(r);paths.append(p);allocated+=s.st_blocks*512
 helpers=active_check(paths)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   args=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
   assert str(ROOT) not in args
   assert not any(s in args for s in ['isaacsim','omni.isaac','headless.kit']), 'Simulator process still active'
   assert not os.readlink(proc/'cwd').startswith(str(ROOT))
  except (OSError,ProcessLookupError):pass
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'DRY_RUN_PASSED',
  'mode':'EXECUTE' if a.execute else 'DRY_RUN','allocated_bytes':allocated,'files_verified':len(rows),
  'disk_before':shutil.disk_usage('/home/ouyanghao')._asdict(),'removed':[],'datasets_deleted':False,
  'known_os_helpers':helpers}
 def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
 save()
 if a.execute:
  active_check(paths)
  for r in rows:
   p,s=check(r);p.unlink();result['removed'].append({'path':str(p),'bytes':r['bytes'],'sha256':r['sha256'],'archive_member':r['archive_member']});save()
  result['status']='BACKED_UP_BUILTIN_TEXTURE_CACHES_REMOVED'
 result['disk_after']=shutil.disk_usage('/home/ouyanghao')._asdict();save();print(json.dumps(result))

if __name__=='__main__':main()
