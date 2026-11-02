"""Unique fresh formal evaluator dispatch, without changing generator jobs."""
import ast
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B=pathlib.Path(__file__).resolve().parents[1]
spec=B/'configs/tmr69d-evaluator-train-spec.json'
s=json.loads(spec.read_bytes())
assert s['formal_training_authorized'] and s['epochs']==500
out=B/'results/semantic_evaluator_20261029'/s['run']
out.mkdir(exist_ok=False)
code=r'''
import os,sys,pathlib,json,fcntl,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import owned,read,sha
from contact_filter_resources import sample_resources
sp=owned(r/'configs/tmr69d-evaluator-train-spec.json');assert sha(sp)==SPEC_SHA_LITERAL;s=read(sp);run=s['run'];prefix='tmr_evaluator_train_'+run
parent=owned(r/'evaluator_tmr_20261029');assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2+16*1024**2)
ck=owned(s['checkpoint_dir'],pathlib.Path('/home/ouyanghao'));assert not list(ck.iterdir())
assert shutil.disk_usage(ck).free>=s['reserve_bytes']+s['home_log_allowance_bytes']+3*s['checkpoint_upper_bytes']
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2;assert gpu>=20*1024**3
assert sha(r/'scripts/train_tmr69d_evaluator.py')==s['executor_sha256'] and sha(r/'scripts/supervise_tmr69d_evaluator_training.py')==s['supervisor_sha256']
for name,value in s['bound_sources'].items():assert sha(owned(r/name))==value
for item in s['witness_bindings']:assert sha(parent/item['run']/'receipt.json')==item['sha256']['receipt']
driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
cmd=['python3',str(r/'scripts/supervise_tmr69d_evaluator_training.py'),'--request',str(sp)]
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
lock.close()
with (parent/(run+'_outer.log')).open('xb') as log:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
proof=dict(status='FRESH_TMR69D_FORMAL_TRAINING_DISPATCHED_NOT_YET_ACCEPTED',run=run,driver_pid=q.pid,spec_sha256=sha(sp),runtime_binding_sha256=s['runtime_binding_sha256'],resources=resources,gpu1_free_bytes=gpu,home_free_bytes=shutil.disk_usage(ck).free)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''.replace('SPEC_SHA_LITERAL',repr(hashlib.sha256(spec.read_bytes()).hexdigest()))
ast.parse(code,feature_version=(3,8))
j=json.loads(recorded_remote(code,out/'dispatch_client.json',timeout=180))
(out/'dispatch.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
