"""Inventory or clear only fixed user-owned derived graphics caches.

Keep logs, datasets, checkpoints, source code, environments and all user data.
No recursive delete is used. Abort if any process maps/opens a selected file.
"""
import datetime,hashlib,json,os,pathlib,shutil,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh')
CACHE=ROOT/'conditional-human-motion-generation/cache'
REL=[
 'r3_official_kit/cache/nv_shadercache','r3_official_kit/cache/DerivedDataCache',
 'r3_punch_viewer_kit/cache/nv_shadercache','r3_punch_viewer_kit/cache/DerivedDataCache',
 'r3_i1_live_viewer/kit/cache/nv_shadercache','r3_i1_live_viewer/kit/cache/DerivedDataCache',
 'r3_i1_viewer_stream/kit/cache/nv_shadercache','r3_i1_viewer_stream/kit/cache/DerivedDataCache',
 'r3_xdg/ov/texturecache','r3_i1_live_viewer/xdg/ov/texturecache']
OUT=ROOT/'textop-reproduction-20261024/receipts'

def inside(p,root):
 return root in p.parents

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def main(apply=False):
 assert os.getuid()==1013 and ROOT.resolve()==ROOT
 folders=[(CACHE/x).resolve() for x in REL]
 assert all(inside(p,CACHE) and p.stat().st_uid==1013 for p in folders if p.exists())
 files=[]
 for folder in folders:
  if not folder.exists():continue
  assert not folder.is_symlink()
  for p in folder.rglob('*'):
   assert not p.is_symlink() and inside(p.resolve(),folder) and p.stat().st_uid==1013
   if not p.is_file():continue
   assert p.suffix.lower() not in ['.log','.py','.pth','.pt','.npz','.pkl','.csv','.mp4','.usd','.urdf']
   files.append(p)
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=1013:continue
   mapped=(proc/'maps').read_text(errors='replace')
   assert not any(str(x)+'/' in mapped for x in folders),'A process maps cache files'
   for fd in (proc/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert not any(target.startswith(str(x)+'/') for x in folders),'A process has cache files open'
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 rows=[{'path':str(p.relative_to(CACHE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
 before=shutil.disk_usage(ROOT).free
 if apply:
  prior=json.loads((OUT/'render_cache_inventory.json').read_text())
  assert prior['files']==rows,'Cache changed since backed-up inventory'
  for p,row in zip(files,rows):
   assert inside(p.resolve(),CACHE) and p.stat().st_uid==1013
   assert sha(p)==row['sha256'];p.unlink()
 result={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'status':'INACTIVE_DERIVED_CACHE_CLEARED' if apply else 'READ_ONLY_INVENTORY',
         'scope':'fixed individual-user shader, derived-data and texture caches only',
         'files':rows,'file_count':len(rows),'logical_bytes':sum(x['bytes'] for x in rows),
         'files_deleted':len(rows) if apply else 0,'data_free_before':before,'data_free_after':shutil.disk_usage(ROOT).free,
         'datasets_deleted':0,'checkpoints_deleted':0,'logs_deleted':0,'environments_deleted':0}
 (OUT/('render_cache_cleanup.json' if apply else 'render_cache_inventory.json')).write_text(json.dumps(result,indent=2))
 return result

if __name__=='__main__':
 r=main('--apply' in sys.argv);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
