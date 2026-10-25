"""Persistent personal-directory and free-space guards, with no file deletion."""
import os,pathlib,shutil,subprocess

def check(destination,allowed_root,policy,remaining_saves=1):
 """Require space for all remaining bounded saves, reserve and log allowance.

The bound is conservative planning, not an exact serialized checkpoint size.
Actual bytes must be checked after each save. Other users may consume space
after this check, so an I/O failure still propagates and retains its partial.
"""
 path=pathlib.Path(destination).absolute();root=pathlib.Path(allowed_root).absolute()
 assert path.is_dir() and root.is_dir(),'Checkpoint destination must already exist'
 assert path==path.resolve() and root==root.resolve(),'Symlinked checkpoint destinations are not allowed'
 assert path==root or root in path.parents,'Checkpoint destination outside personal task root'
 assert type(remaining_saves) is int and remaining_saves>=1
 here=path
 while True:
  assert here.stat().st_uid==os.getuid()==1013,'Destination ownership changed'
  if here==root:break
  here=here.parent
 assert os.access(path,os.W_OK|os.X_OK),'Checkpoint destination is not writable'
 fs=subprocess.check_output(['findmnt','-T',str(path),'-n','-o','FSTYPE'],text=True).strip()
 assert fs in ('ext4','xfs','btrfs','zfs'),'Checkpoint destination must use a persistent disk filesystem'
 reserve=int(policy['reserve_bytes']);logs=int(policy['log_allowance_bytes']);bound=int(policy['checkpoint_upper_bound_bytes'])
 assert reserve>=2*1024**3 and logs>=512*1024**2 and bound>=1
 free=shutil.disk_usage(path).free;required=reserve+logs+remaining_saves*bound
 assert free>=required,'Insufficient checkpoint capacity: free=%d required=%d; no checkpoint written'%(free,required)
 return {'destination':str(path),'filesystem':fs,'owner_uid':os.getuid(),'free_bytes':free,
         'required_bytes':required,'remaining_saves':remaining_saves,'checkpoint_upper_bound_bytes':bound,
         'reserve_bytes':reserve,'log_allowance_bytes':logs,'persistent_owned_writable':True}

def schedule(target,interval):
 assert target==40000 and interval==5000
 return [1000]+list(range(interval,target+1,interval))
