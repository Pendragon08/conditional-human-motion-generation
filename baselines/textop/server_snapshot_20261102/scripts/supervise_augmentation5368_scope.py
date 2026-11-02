"""Metadata-only scope supervisor, independent of the active model lock.

Only standard-library metadata is permitted. This supervisor never signals,
imports, or acquires the lock of a production model job.
"""
import argparse
import fcntl
import json
import os
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from filtered15000_vae_completion import ROOT,owned,read,sha

p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
sp=owned(a.request);s=read(sp)
assert sha(__file__)==s['supervisor_sha256'] and os.getuid()==1013
assert sha(ROOT/'scripts/build_augmentation5368_scope.py')==s['executor_sha256']
assert s['scope_authorized'] and not s['generation_authorized'] and not s['ppo_train_authorized']
parent=ROOT/'step34_20261029';parent.mkdir(mode=0o700,exist_ok=True);owned(parent)
lock=(ROOT/'receipts/user_step34_metadata.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (parent/s['run']).exists()
assert shutil.disk_usage(parent).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['metadata_upper_bytes']
log=parent/(s['run']+'_supervision');log.mkdir(mode=0o700,exist_ok=False)
def sample(pid):
    return sample_resources(ROOT,pid,16*1024**3,s['reserve_bytes']+s['log_allowance_bytes'])
command=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONIOENCODING=utf-8',
         'PYTHONDONTWRITEBYTECODE=1','/usr/bin/python3','-B',
         str(ROOT/'scripts/build_augmentation5368_scope.py'),'--request',str(sp)]
proof=run_supervised(command,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,900,15)
rp=parent/s['run']/'receipt.json';receipt=read(rp) if rp.exists() else {}
proof.update(run=s['run'],source_sha256=sha(__file__),request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None,resources_sha256=sha(log/'resources.jsonl'),
    process_exit_sha256=sha(log/'process_exit.json'),model_lock_acquired=False)
with (log/'exit.json').open('x') as stream:
    json.dump(proof,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
assert proof['supervision_passed'] and receipt['status']=='USER_STEP3_AUGMENTATION5368_METADATA_SCOPE_BUILT'
assert receipt['output_writers_closed'] and receipt['actual_models_loaded']==receipt['actual_motion_arrays_read']==0
assert receipt['actual_GPU_contexts']==receipt['actual_generated_clips']==receipt['actual_PPO_updates']==0
