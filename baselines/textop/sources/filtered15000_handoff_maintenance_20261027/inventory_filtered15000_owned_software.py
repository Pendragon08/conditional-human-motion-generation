"""Bounded metadata-only list of this task's owned software directories."""
import ast,json,pathlib,sys
T=pathlib.Path(__file__).resolve().parent
B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'))
from backup_filter_recovery_checkpoint import recorded_remote
E=B/'results/filtered15000_handoff_20261027/storage_recovered_20261028_v1'
code=r'''
import json,os,pathlib,subprocess,time
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');h=pathlib.Path('/home/ouyanghao')
items=[]
for parent in [r,h]:
 assert parent.resolve()==parent and parent.stat().st_uid==1013
 for p in sorted(parent.iterdir()):
  st=p.lstat()
  if st.st_uid!=1013 or p.is_symlink():continue
  row=dict(path=str(p),is_dir=p.is_dir(),owner_uid=st.st_uid,device=st.st_dev)
  if p.is_file():row['bytes']=st.st_size
  items.append(row)
software=[]
# Named software-only candidates; no data/checkpoints/env active in training.
for name in ['software','third_party','archives','downloads','src','source','build']:
 p=r/name
 if p.is_dir() and not p.is_symlink() and p.stat().st_uid==1013:
  try:
   v=subprocess.run(['du','-sx','--block-size=1','--',str(p)],capture_output=True,text=True,timeout=20)
   software.append(dict(path=str(p),actual_exit_code=v.returncode,allocated_bytes=int(v.stdout.split()[0]) if v.returncode==0 else None))
  except subprocess.TimeoutExpired:software.append(dict(path=str(p),metadata_size_scan_timed_out=True))
print(json.dumps(dict(observed_unix=time.time(),top_level_owned_entries=items,software_candidates=software,
 files_deleted=0,recovery_provenance_verified=False,cleanup_authorized_by_this_inventory=False,new_model_loads=0,new_gpu_contexts=0,
 note='Owned top-level metadata only. This list does not authorize deletion; active env and all arrays/checkpoints excluded from software size scans.')))
'''
ast.parse(code)
j=json.loads(recorded_remote(code,E/'software_inventory_client.json',timeout=150))
(E/'software_inventory.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
