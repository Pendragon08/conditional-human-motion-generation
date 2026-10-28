"""Actual-scope witness activation, then two-real-witness diagnostic activation."""
import argparse
import ast
import base64
import hashlib
import json
import pathlib
import sys
from backup_filter_recovery_checkpoint import recorded_remote
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_generation_acceptance import RUNS, runtime_signature, validate_witness

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['witnesses', 'diagnostic'], required=True)
p.add_argument('--document-summary')
a = p.parse_args()
sp = B / 'configs/ldm100k-generation-spec.json'
old = sp.read_bytes()
s = json.loads(old)
assert not s['generation_authorized'] and s['witness_bindings'] is None
doc64 = None
if a.stage == 'witnesses':
    assert hashlib.sha256(old).hexdigest() == 'f2e1185d32aa599b71d1c2dca68d41ef6503bbbbff8d3b51fe1b73b9fb45e229'
    assert not s['witnesses_authorized'] and s['scope_binding'] is None
    j = json.loads((O / 'user_ldm100k_heldout_scope_v1/progress.json').read_bytes())
    assert j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
    assert j['files']['receipt']['status'] == 'USER_LDM100K_HELDOUT_METADATA_SCOPE_BUILT'
    s['scope_binding'] = dict(run=j['run'], receipt_sha256=j['sha256']['receipt'],
        exit_sha256=j['sha256']['exit'], spec_sha256=j['files']['receipt']['spec_sha256'], closure=j['closure'])
    s['bound_sources']['scripts/ldm100k_generation_artifacts.py'] = hashlib.sha256(
        (B / 'scripts/ldm100k_generation_artifacts.py').read_bytes()).hexdigest()
else:
    assert s['witnesses_authorized'] and s['scope_binding'] and a.document_summary
    docpath = pathlib.Path(a.document_summary).resolve()
    assert docpath.name == 'SUMMARY.json'
    guard_local(B.parents[3] / 'textop_reproduction_20261024', docpath)
    raw = docpath.read_bytes()
    document = json.loads(raw)
    runs = {}
    for run in RUNS:
        j = json.loads((O / run / 'progress.json').read_bytes())
        assert j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
        assert j['artifact_audit']['real_numpy_array_payloads_checked'] == 10
        r = j['files']['receipt']
        validate_witness(r, run, s)
        assert r['spec_sha256'] == hashlib.sha256(old).hexdigest()
        runs[run] = dict(receipt_sha256=j['sha256']['receipt'], exit_sha256=j['sha256']['exit'], closure=j['closure'])
    s['witness_bindings'] = dict(witness_spec_sha256=hashlib.sha256(old).hexdigest(), runs=runs,
        runtime_binding_sha256=runtime_signature(s),
        document_path='/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/ldm100k_generation_fresh_document_verified.json',
        document_sha256=hashlib.sha256(raw).hexdigest())
    assert document['status'] == 'USER_LDM100K_GENERATION_FRESH_DOCUMENT_VERIFIED'
    assert document['run'] == RUNS[1] and type(document['actual_exit_code']) is int and document['actual_exit_code'] == 0
    assert document['findings'] == [] and document['source_sha256'] == s['executor_sha256']
    assert document['runtime_binding_sha256'] == s['witness_bindings']['runtime_binding_sha256']
    doc64 = base64.b64encode(raw).decode()
code = r'''
import base64,fcntl,hashlib,json,os,pathlib,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned,closed_job
from ldm100k_scope_acceptance import accepted_scope
from ldm100k_generation_protocol import pair_output_upper_bytes
from ldm100k_generation_acceptance import RUNS,validate_witness,accepted_generation_witnesses,runtime_signature
from ldm100k_generation_artifacts import audit_witness_artifacts
from contact_filter_resources import sample_resources
stage=STAGE;s=POLICY;p=owned(r/'configs/ldm100k-generation-spec.json');assert sha(p)==OLD_SHA
assert not read(p)['generation_authorized'];lock=(r/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,h in s['bound_sources'].items():assert sha(owned(r/n))==h,n
assert sha(r/'scripts/run_ldm100k_heldout_generation.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_heldout_generation.py')==s['supervisor_sha256']
actual=accepted_scope(s['scope_binding']);pending_document=None
if stage=='witnesses':
 assert not read(p)['witnesses_authorized']
 first=actual['scope']['segments'][0]
 s['accepted_output_upper_bytes']=dict(witness=pair_output_upper_bytes(first['future_frames']),diagnostic=actual['plan']['output_upper_bound_bytes'])
 s['witnesses_authorized']=True
 archive=r/'receipts/ldm100k_generation_spec_before_witness.json'
else:
 assert read(p)['witnesses_authorized'] and read(p)['scope_binding']==s['scope_binding']
 for run in RUNS:
  rp=owned(r/'ldm100k_user_handoff_20261028'/run/'receipt.json');rec=read(rp)
  assert sha(rp)==s['witness_bindings']['runs'][run]['receipt_sha256'];validate_witness(rec,run,s)
  assert rec['spec_sha256']==OLD_SHA
  proof=closed_job('ldm100k_generation_'+run,rp,sha(rp),s['supervisor_sha256'])
  assert proof==s['witness_bindings']['runs'][run]['closure']
  assert proof['exit_sha256']==s['witness_bindings']['runs'][run]['exit_sha256']
  audit_witness_artifacts(rp.parent,rec)
 raw=base64.b64decode(__DOCUMENT_BYTES__);v=s['witness_bindings'];assert hashlib.sha256(raw).hexdigest()==v['document_sha256']
 dp=pathlib.Path(v['document_path']);assert dp.parent==r/'receipts' and not dp.exists()
 doc=json.loads(raw);assert doc['run']==RUNS[1] and doc['status']=='USER_LDM100K_GENERATION_FRESH_DOCUMENT_VERIFIED'
 assert type(doc['actual_exit_code']) is int and doc['actual_exit_code']==0 and doc['findings']==[]
 assert doc['source_sha256']==s['executor_sha256'] and doc['runtime_binding_sha256']==runtime_signature(s)==v['runtime_binding_sha256']
 assert doc['actual_witness_receipt_sha256']==v['runs'][RUNS[1]]['receipt_sha256']
 assert doc['actual_witness_pair_arrays']==rec['witness_pair_arrays'] and doc['actual_witness_pair_probe_sha256']==rec['witness_pair_probe_sha256']
 pending_document=(dp,raw)
 archive=r/'receipts/ldm100k_generation_spec_before_diagnostic.json'
resources=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes'])
mode='witness' if stage=='witnesses' else 'diagnostic';budget=s['accepted_output_upper_bytes'][mode]
assert __import__('shutil').disk_usage(r).free>=s['reserve_bytes']+s['log_allowance_bytes']+budget
import subprocess
gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip())*1024**2>=20*1024**3
with archive.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
if pending_document is not None:
 dp,document_bytes=pending_document
 with dp.open('xb') as f:os.chmod(dp,0o600);f.write(document_bytes);f.flush();os.fsync(f.fileno())
if stage=='diagnostic':accepted_generation_witnesses(s['witness_bindings']);s['witnesses_authorized']=False;s['generation_authorized']=True
raw=(json.dumps(s,indent=2)+'\n').encode();temp=p.with_name(p.name+'.'+stage+'.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='LDM100K_GENERATION_'+stage.upper()+'_ENABLED_NOT_DISPATCHED',stage=stage,new_spec_sha256=sha(p),
 new_spec_base64=base64.b64encode(raw).decode(),preflight_resources=resources,capacity_checked_for_mode=mode,
 accepted_output_upper_bytes=s['accepted_output_upper_bytes'],actual_model_loads=0,new_gpu_contexts=0,actual_pair_generation=0)
with (r/'receipts'/('ldm100k_generation_'+stage+'_activation.json')).open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('STAGE', repr(a.stage)).replace('POLICY', repr(s)).replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('__DOCUMENT_BYTES__', repr(doc64))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / ('generation_' + a.stage + '_activate_client.json'), timeout=180))
raw = base64.b64decode(z['new_spec_base64'])
assert hashlib.sha256(raw).hexdigest() == z['new_spec_sha256']
with (O / ('generation_spec_before_' + a.stage + '.json')).open('xb') as f:
    f.write(old)
sp.write_bytes(raw)
(O / ('generation_' + a.stage + '_activation.json')).write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
