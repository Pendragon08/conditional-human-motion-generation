"""Bind a genuine closed independent100k verifier and its real document summary."""
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
from ldm100k_verified_input import validate_document
from ldm100k_scope_contract import validate_policy

p = argparse.ArgumentParser()
p.add_argument('--document-summary', required=True)
a = p.parse_args()
document = pathlib.Path(a.document_summary).resolve()
assert document.is_file() and document.name == 'SUMMARY.json'
guard_local(B.parents[3] / 'textop_reproduction_20261024', document)
raw_document = document.read_bytes()
j = json.loads((O / 'user_ldm100k_prefix_document_v1/progress.json').read_bytes())
assert j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
rec = j['files']['receipt']
assert rec['status'] == 'USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED'
sp = B / 'configs/ldm100k-heldout-scope-spec.json'
old = sp.read_bytes()
s = json.loads(old)
validate_policy(s, enabled=False)
s['verified_model'] = dict(run='user_ldm100k_prefix_document_v1',
    receipt_sha256=j['sha256']['receipt'], exit_sha256=j['sha256']['exit'],
    spec_sha256=rec['spec_sha256'], closure=j['closure'],
    document_path='/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/ldm100k_prefix_fresh_document_verified.json',
    document_sha256=hashlib.sha256(raw_document).hexdigest())
validate_document(json.loads(raw_document), rec, s['verified_model'])
code = r'''
import base64,fcntl,json,os,pathlib,shutil,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned,closed_job,validate_document,accepted_verified_input
from ldm100k_scope_contract import validate_policy
from ldm100k_export_acceptance import accepted_export
from contact_filter_resources import sample_resources
p=owned(r/'configs/ldm100k-heldout-scope-spec.json');assert sha(p)==OLD_SHA;s=POLICY
assert not read(p)['scope_build_authorized'];lock=(r/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert sha(r/'scripts/build_ldm100k_heldout_scope.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_heldout_scope.py')==s['supervisor_sha256']
v=s['verified_model'];rp=owned(r/'ldm100k_user_handoff_20261028'/v['run']/'receipt.json');rec=read(rp)
assert sha(rp)==v['receipt_sha256'];vs=owned(r/'configs/ldm100k-prefix-verify-spec.json')
assert sha(vs)==v['spec_sha256'];verify_spec=read(vs)
proof=closed_job('ldm100k_prefix_verify_'+v['run'],rp,sha(rp),verify_spec['supervisor_sha256'])
assert proof==v['closure'] and proof['exit_sha256']==v['exit_sha256']
accepted_export(verify_spec['export_binding'])
raw=base64.b64decode(DOCUMENT);assert __import__('hashlib').sha256(raw).hexdigest()==v['document_sha256']
validate_document(json.loads(raw),rec,v);dp=pathlib.Path(v['document_path'])
assert dp.parent==r/'receipts' and not dp.exists();owned(dp.parent)
with dp.open('xb') as f:os.chmod(dp,0o600);f.write(raw);f.flush();os.fsync(f.fileno())
actual=accepted_verified_input(v)
resources=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes'])
assert sha(r/'configs/filtered15000-corpus-spec.json')==s['corpus_spec_sha256']
assert sha(r/'configs/ldm100k-heldout-plan.json')==s['heldout_plan_sha256']
archive=r/'receipts/ldm100k_heldout_scope_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
s['scope_build_authorized']=True;validate_policy(s);new=(json.dumps(s,indent=2)+'\n').encode()
temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(new);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='LDM100K_HELDOUT_SCOPE_ENABLED_NOT_DISPATCHED',new_spec_sha256=sha(p),
 new_spec_base64=base64.b64encode(new).decode(),verified_model_receipt_sha256=actual['receipt_sha256'],
 preflight_resources=resources,actual_scope_builds=0,actual_model_loads=0,actual_gpu_contexts=0)
with (r/'receipts/ldm100k_heldout_scope_activation.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('POLICY', repr(s)).replace('DOCUMENT', repr(base64.b64encode(raw_document).decode()))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'scope_activate_client.json', timeout=180))
new = base64.b64decode(z['new_spec_base64'])
assert hashlib.sha256(new).hexdigest() == z['new_spec_sha256']
with (O / 'scope_spec_before_enable.json').open('xb') as handle:
    handle.write(old)
sp.write_bytes(new)
(O / 'scope_activation.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
