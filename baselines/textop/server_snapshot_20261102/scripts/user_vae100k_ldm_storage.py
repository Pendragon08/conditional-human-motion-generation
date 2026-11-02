"""Two owned persistent disks; full states retained,2GiB reserve on EACH disk."""
import hashlib,json,os,pathlib,shutil,subprocess
STEPS=[100000,200000,300000]

def directory(spec,step=None,emergency=False):
 plan=spec['checkpoint_layout'];name=plan['slots']['emergency' if emergency else str(step)]
 item=plan['volumes'][name]
 return pathlib.Path(item['checkpoint_dir']),pathlib.Path(item['allowed_root'])

def check(spec,pending_steps=None,reserve_emergency=True):
 pending=STEPS if pending_steps is None else list(pending_steps)
 assert len(pending)==len(set(pending)) and all(x in STEPS for x in pending)
 plan=spec['checkpoint_layout'];policy=spec['checkpoint_policy']
 assert plan['schema']=='textop.user_vae100k_ldm_dual_storage.v1'
 assert plan['slots']=={'100000':'home','200000':'data','300000':'home','emergency':'data'}
 assert set(plan['volumes'])=={'home','data'}
 assert plan['volumes']['home']['log_allowance_bytes']==0
 assert plan['volumes']['data']['log_allowance_bytes']==policy['log_allowance_bytes']>=512*1024**2
 assert policy['reserve_bytes']>=2*1024**3 and policy['checkpoint_upper_bound_bytes']==300*1024**2
 assert spec['checkpoint_dir']==plan['volumes']['home']['checkpoint_dir'] and spec['checkpoint_allowed_root']==plan['volumes']['home']['allowed_root']
 devices=set();observed=[]
 for name,item in plan['volumes'].items():
  dest=pathlib.Path(item['checkpoint_dir']);root=pathlib.Path(item['allowed_root'])
  assert dest.is_dir() and root.is_dir() and dest.resolve()==dest and root.resolve()==root and root in dest.parents
  here=dest
  while True:
   assert here.lstat().st_uid==os.getuid()==1013 and not here.is_symlink()
   if here==root:break
   here=here.parent
  assert os.access(dest,os.W_OK|os.X_OK)
  fs=subprocess.check_output(['findmnt','-T',str(dest),'-n','-o','FSTYPE'],text=True,timeout=15).strip();assert fs in ['ext4','xfs','btrfs','zfs']
  device=dest.stat().st_dev;assert device not in devices,'Do not double-count one filesystem';devices.add(device)
  slots=sum(plan['slots'][str(x)]==name for x in pending)+int(reserve_emergency and plan['slots']['emergency']==name)
  cache=spec['vae_cache_upper_bound_bytes'] if name=='home' and not (dest/'vae.pth').exists() else 0
  required=policy['reserve_bytes']+item['log_allowance_bytes']+slots*policy['checkpoint_upper_bound_bytes']+cache;free=shutil.disk_usage(dest).free
  assert free>=required,'Insufficient actual '+name+' capacity: free='+str(free)+' required='+str(required)
  observed.append(dict(volume=name,destination=str(dest),filesystem=fs,device=device,owner_uid=1013,free_bytes=free,required_bytes=required,remaining_slots=slots,cache_budget_bytes=cache,reserve_bytes=policy['reserve_bytes'],log_allowance_bytes=item['log_allowance_bytes']))
 return dict(layout_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),volumes=observed,pending_steps=pending,reserve_emergency=reserve_emergency,global_log_budget_bytes=policy['log_allowance_bytes'],reserve_on_each_filesystem_bytes=policy['reserve_bytes'])

def log_allowance(spec,emergency=False,step=None):
 name=spec['checkpoint_layout']['slots']['emergency' if emergency else str(step)]
 return spec['checkpoint_layout']['volumes'][name]['log_allowance_bytes']

def contains_state_files(spec):
 return [str(p) for item in spec['checkpoint_layout']['volumes'].values() for pattern in ['ldm_state_*.pth','emergency_*.pth','*.partial'] for p in pathlib.Path(item['checkpoint_dir']).glob(pattern)]
