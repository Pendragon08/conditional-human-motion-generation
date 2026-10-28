"""Actual metadata-only user Step3 scope; no model, ndarray or SDK imports."""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from filtered15000_vae_completion import ROOT,owned,read,sha
from augmentation5368_protocol import build_plan

p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
sp=owned(a.request);s=read(sp)
assert os.getuid()==1013 and sha(__file__)==s['executor_sha256'] and s['scope_authorized'] is True
assert s['generation_authorized'] is False and s['ppo_train_authorized'] is False
for name,value in s['bound_sources'].items():assert sha(owned(ROOT/name))==value
item=s['train_scope_receipt'];rp=owned(item['path']);assert sha(rp)==item['sha256']
receipt=read(rp)
assert receipt['status']=='TMR69D_TRAINING_METADATA_SCOPE_BUILT' and receipt['output_writers_closed']
item=receipt['private_artifact'];source=owned(item['path'])
assert source.stat().st_size==item['bytes'] and sha(source)==item['sha256']
data=read(source)
parent=ROOT/'step34_20261029';parent.mkdir(mode=0o700,exist_ok=True);owned(parent)
out=parent/s['run'];out.mkdir(mode=0o700,exist_ok=False)
started=time.monotonic();plan=build_plan(data)
raw=(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
assert len(raw)<=s['metadata_upper_bytes']
assert shutil.disk_usage(out).free>=s['reserve_bytes']+s['log_allowance_bytes']+len(raw)
path=out/'augmentation_scope.private.json'
with path.open('xb') as stream:
    os.chmod(path,0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sha(source)==item['sha256']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'isaaclab' not in sys.modules
result=dict(status='USER_STEP3_AUGMENTATION5368_METADATA_SCOPE_BUILT',run=s['run'],source_sha256=sha(__file__),
    request_sha256=sha(sp),train_scope_receipt_sha256=sha(rp),private_artifact=dict(path=str(path),bytes=len(raw),sha256=sha(path)),
    summary=plan['summary'],actual_models_loaded=0,actual_motion_arrays_read=0,actual_GPU_contexts=0,
    actual_generated_clips=0,actual_PPO_updates=0,actual_new_SDK_tasks=0,output_writers_closed=True,
    elapsed_seconds=time.monotonic()-started)
with (out/'receipt.json').open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
print(json.dumps(result))
