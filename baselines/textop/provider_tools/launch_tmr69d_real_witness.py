"""Unique real-data GPU witness dispatch, gated by actual closed inputs."""
import argparse
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B/'results/semantic_evaluator_20261029'
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
p = argparse.ArgumentParser()
p.add_argument('--role', choices=['author','document'], required=True)
a = p.parse_args()
run = 'tmr69d_real_witness_' + a.role + '_v1'
roles = {key: json.loads((O/name/'progress.json').read_bytes()) for key,name in [
    ('scope','tmr69d_training_scope_v2'),('numeric','tmr69d_numeric_data_v1'),('text','tmr_text_models_import_verify_v1')]}
assert all(x['closed_execution_verified'] and not x['workers'] for x in roles.values())
inputs = {key+'_receipt':dict(path=ROOT+'/evaluator_tmr_20261029/'+value['run']+'/receipt.json',sha256=value['sha256']['receipt']) for key,value in roles.items()}
uploaded = json.loads((O/'public_text_upload_v1/receipt.json').read_bytes())
inputs['import_receipt'] = dict(path=uploaded['receipt_path'],sha256=uploaded['receipt_sha256'])
inputs['codec_sha256'] = hashlib.sha256((B/'scripts/paper69_runtime.py').read_bytes()).hexdigest()
upstream = {name:hashlib.sha256((B/'sources/tmr_upstream_20261029/src/model'/(name+'.py')).read_bytes()).hexdigest() for name in ['actor','losses','metrics','temos','tmr']}
sources = {}
for name in ['scripts/run_tmr69d_real_witness.py','scripts/supervise_tmr69d_real_witness.py','scripts/tmr69d_training_inputs.py']:
    raw = (B/name).read_bytes()
    ast.parse(raw, feature_version=(3,8))
    sources[name] = dict(raw=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
bound = {name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in ['scripts/tmr69d_core.py','scripts/paper69_runtime.py','scripts/generator_training_state.py']}
bound['scripts/tmr69d_training_inputs.py'] = sources['scripts/tmr69d_training_inputs.py']['sha256']
policy = dict(inputs=inputs,bound_sources=bound,upstream_model_sha256=upstream,seed=20261029,batch_size=32,
    upstream_model_dir=ROOT+'/evaluator_tmr_20261029/upstream_core_v1',formal_training_authorized=False)
binding = hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
s = dict(policy,run=run,role=a.role,runtime_binding_sha256=binding,
    executor_sha256=sources['scripts/run_tmr69d_real_witness.py']['sha256'],supervisor_sha256=sources['scripts/supervise_tmr69d_real_witness.py']['sha256'],
    output_upper_bytes=16*1024**2,reserve_bytes=2*1024**3,log_allowance_bytes=128*1024**2)
author = None
if a.role == 'document':
    author = json.loads((O/'tmr69d_real_witness_author_v1/progress.json').read_bytes())
    assert author['closed_execution_verified'] and author['artifact_audit']['real_data_fullstate_replay_and_zero_validation_updates_verified']
    assert author['files']['receipt']['runtime_binding_sha256'] == binding
    s['author_receipt'] = dict(path=ROOT+'/evaluator_tmr_20261029/tmr69d_real_witness_author_v1/receipt.json',sha256=author['sha256']['receipt'])
out = O/run
out.mkdir(exist_ok=False)
code = r'''
import os,sys,pathlib,json,base64,hashlib,fcntl,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import owned,read,sha
from contact_filter_resources import sample_resources
s=SPEC_LITERAL;sources=SOURCES_LITERAL;run=s['run'];prefix='tmr_real_witness_'+run;parent=owned(r/'evaluator_tmr_20261029')
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_request.json','_driver.json','_driver.py','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2+s['output_upper_bytes'])
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2;assert gpu>=20*1024**3
for item in [s['inputs'][k+'_receipt'] for k in ['scope','numeric','text']]+[s['inputs']['import_receipt']]:assert sha(owned(item['path']))==item['sha256']
if 'author_receipt' in s:
 item=s['author_receipt'];assert sha(owned(item['path']))==item['sha256'];author=read(item['path']);assert author['status']=='TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED' and author['output_writers_closed'] and author['runtime_binding_sha256']==s['runtime_binding_sha256']
for name,item in sources.items():
 path=r/name
 if path.exists():assert sha(owned(path))==item['sha256']
 else:
  raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
for name,value in s['bound_sources'].items():assert sha(owned(r/name))==value
sp=r/'receipts'/(prefix+'_request.json')
with sp.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
cmd=['python3',str(r/'scripts/supervise_tmr69d_real_witness.py'),'--request',str(sp)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
lock.close()
with (parent/(run+'_outer.log')).open('xb') as log:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
proof=dict(status='REAL_TMR_GPU_WITNESS_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(sp),runtime_binding_sha256=s['runtime_binding_sha256'],gpu1_free_bytes=gpu,resources=resources)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''.replace('SPEC_LITERAL',repr(s)).replace('SOURCES_LITERAL',repr(sources))
ast.parse(code,feature_version=(3,8))
result=json.loads(recorded_remote(code,out/'dispatch_client.json',timeout=180))
(out/'dispatch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
