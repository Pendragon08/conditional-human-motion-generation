"""Enable only independent CPU verification from the actual closed CPU export."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
j = json.loads((O / 'user_ldm100k_prefix_export_v1/progress.json').read_bytes())
assert j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
r = j['files']['receipt']
assert r['status'] == 'USER_LDM100000_PREFIX_CPU_EXPORT_PASSED'
sp = B / 'configs/ldm100k-prefix-verify-spec.json'
old = sp.read_bytes()
s = json.loads(old)
assert not s['verify_authorized'] and s['export_binding'] is None
s['export_binding'] = dict(run='user_ldm100k_prefix_export_v1', receipt_sha256=j['sha256']['receipt'],
                           exit_sha256=j['sha256']['exit'], spec_sha256=r['spec_sha256'], inference=r['inference'])
code = r'''
import base64,fcntl,json,os,pathlib,shutil,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(r/'scripts'))
from ldm100k_export_acceptance import accepted_export,read,sha
p=r/'configs/ldm100k-prefix-verify-spec.json';assert sha(p)==OLD_SHA;s=POLICY
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
accepted=accepted_export(s['export_binding'])
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert sha(r/'scripts/verify_ldm100k_prefix_export.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_prefix_verify.py')==s['supervisor_sha256']
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>=s['minimum_ram_free_bytes'] and shutil.disk_usage(r).free>=s['minimum_disk_free_bytes']
assert shutil.disk_usage(accepted['inference']['path']).free>=s['reserve_bytes']
archive=r/'receipts/ldm100k_prefix_verify_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
s['verify_authorized']=True;raw=(json.dumps(s,indent=2)+'\n').encode();temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(p)
z=dict(status='LDM100K_PREFIX_CPU_VERIFICATION_ENABLED_NOT_DISPATCHED',new_spec_sha256=sha(p),
 new_spec_base64=base64.b64encode(raw).decode(),actual_model_loads=0,actual_gpu_contexts=0,actual_verifications=0,
 export_receipt_sha256=accepted['receipt_sha256'])
with (r/'receipts/ldm100k_prefix_verify_activation.json').open('x') as f:json.dump(z,f,indent=2)
lock.close();print(json.dumps(z))
'''.replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('POLICY', repr(s))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'prefix_verify_activate_client.json', timeout=120))
raw = base64.b64decode(z['new_spec_base64'])
assert hashlib.sha256(raw).hexdigest() == z['new_spec_sha256']
with (O / 'prefix_verify_spec_before_enable.json').open('xb') as handle:
    handle.write(old)
sp.write_bytes(raw)
(O / 'prefix_verify_activation.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
