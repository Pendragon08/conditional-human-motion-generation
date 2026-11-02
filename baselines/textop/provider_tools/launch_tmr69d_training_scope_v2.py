"""Build an actual TRAIN-only semantic data scope from accepted private metadata."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/semantic_evaluator_20261029'
RUN = 'tmr69d_training_scope_v2'
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
manifest = O / 'public_dependency_manifest_v2.json'
m = json.loads(manifest.read_bytes())
assert len(m['packages']) == 6 and m['total_selected_wheel_bytes'] == 16649167
fidelity = json.loads((B / 'results/ldm100k_user_handoff_20261028/user_ldm100k_tracking_fidelity_v1/progress.json').read_bytes())
assert fidelity['closed_execution_verified'] and fidelity['artifact_audit']['threshold_and_aggregate_recomputed']
sources = {}
for name in ['scripts/build_tmr69d_training_scope_v2.py', 'scripts/supervise_tmr69d_training_scope_v2.py']:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 8))
    sources[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
heldout = json.loads((B / 'results/ldm100k_user_handoff_20261028/user_ldm100k_heldout_scope_v1/progress.json').read_bytes())
assert heldout['closed_execution_verified']
prior = heldout['files']['receipt']
spec = dict(run=RUN, executor_sha256=sources['scripts/build_tmr69d_training_scope_v2.py']['sha256'],
    supervisor_sha256=sources['scripts/supervise_tmr69d_training_scope_v2.py']['sha256'],
    inputs_helper_sha256=hashlib.sha256((B / 'scripts/filtered15000_inputs.py').read_bytes()).hexdigest(),
    corpus_spec_sha256=hashlib.sha256((B / 'configs/filtered15000-corpus-spec.json').read_bytes()).hexdigest(),
    corpus_receipt_sha256=prior['source_corpus_receipt_sha256'], data_acceptance_sha256=prior['data_acceptance_sha256'],
    training_authorized=False, output_upper_bytes=96 * 1024**2,
    reserve_bytes=2 * 1024**3, log_allowance_bytes=128 * 1024**2)
out = O / RUN
out.mkdir(exist_ok=False)
code = r'''
import json,pathlib,hashlib,base64,fcntl,os,sys,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import read,sha,owned
from contact_filter_resources import sample_resources
run=RUN_LITERAL;s=SPEC_LITERAL;prefix='tmr_training_scope_'+run
parent=r/'evaluator_tmr_20261029'
if not parent.exists():parent.mkdir(mode=0o700)
parent=owned(parent)
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_outer_exit.json','_request.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2)
gpu_snapshot=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip()
incoming=parent/'text_models_import_v1';remaining=0
if (incoming/'import_intent.json').exists():
 for item in read(incoming/'import_intent.json')['files']:
  p=incoming/item['relative']
  if not p.exists():p=incoming/(item['relative']+'.partial')
  count=p.stat().st_size if p.exists() else 0;assert 0<=count<=item['bytes'];remaining+=item['bytes']-count
assert shutil.disk_usage(parent).free>s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']+remaining
assert sha(owned(r/'configs/filtered15000-corpus-spec.json'))==s['corpus_spec_sha256']
for name,item in SOURCES_LITERAL.items():
 p=r/name;assert not p.exists();raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(owned(p))==item['sha256']
sp=r/'receipts'/(prefix+'_request.json')
with sp.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
lock.close();cmd=['python3',str(r/'scripts/supervise_tmr69d_training_scope_v2.py'),'--request',str(sp)]
driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='TMR_TRAINING_SCOPE_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(sp),preflight_resources=resources,gpu1_free_used_mib=gpu_snapshot,data_free_bytes=shutil.disk_usage(parent).free,model_or_array_loads_authorized=False,evaluator_training_updates=0)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN_LITERAL', repr(RUN)).replace('SPEC_LITERAL', repr(spec)).replace('SOURCES_LITERAL', repr(sources))
ast.parse(code, feature_version=(3, 8))
(out / 'rendered_source_sha256.json').write_text(json.dumps(dict(sha256=hashlib.sha256(code.encode()).hexdigest()), indent=2), encoding='utf-8')
result = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
