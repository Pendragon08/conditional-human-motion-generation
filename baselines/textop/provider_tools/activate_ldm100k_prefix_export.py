"""Bind actual stopped100k/closed-control proof, then enable only CPU export."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
current = json.loads((O / 'user_ldm100k_control_v1/progress.json').read_bytes())
assert current['closed_execution_verified'] and current['common_lock_free'] and not current['workers']
assert current['files']['receipt']['status'] == 'USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
binding = dict(kind='USER_REQUESTED_LDM100000_PREFIX', selected_step=100000,
               control_run='user_ldm100k_control_v1', control_receipt_sha256=current['sha256']['receipt'],
               control_exit_sha256=current['sha256']['exit'],
               control_spec_sha256=current['files']['receipt']['spec_sha256'])
sp = B / 'configs/ldm100k-prefix-export-spec.json'
old = sp.read_bytes()
s = json.loads(old)
assert not s['export_authorized'] and s['ldm_training'] is None
s['ldm_training'] = binding
code = r'''
import base64,fcntl,json,os,pathlib,shutil,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_prefix_completion import stopped_ldm100k,read,sha,owned
p=r/'configs/ldm100k-prefix-export-spec.json';assert sha(p)==OLD_SHA;s=SPEC
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
v=stopped_ldm100k(s['ldm_training'])
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert sha(r/'scripts/export_ldm100k_prefix.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_prefix_export.py')==s['supervisor_sha256']
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>=s['minimum_ram_free_bytes'] and shutil.disk_usage(r).free>=s['minimum_disk_free_bytes']
parent=pathlib.Path(s['export_dir']);allowed=pathlib.Path(s['export_allowed_root']);owned(allowed,allowed)
assert parent.parent==allowed and not parent.exists() and not parent.is_symlink()
required=s['reserve_bytes']+s['log_allowance_bytes']+s['export_upper_bound_bytes']
assert shutil.disk_usage(allowed).free>=required
fs=subprocess.check_output(['stat','-f','-c','%T',str(allowed)],text=True,timeout=15).strip();assert fs in ['ext2/ext3','ext4']
parent.mkdir(mode=0o700);owned(parent,allowed)
s.update(export_authorized=True,actual_endpoint_accepted=True,actual_tensor_export_accepted=False)
archive=r/'receipts/ldm100k_prefix_export_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
raw=(json.dumps(s,indent=2)+'\n').encode();temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='LDM100K_PREFIX_CPU_EXPORT_ENABLED_NOT_DISPATCHED',new_spec_sha256=sha(p),
 new_spec_base64=base64.b64encode(raw).decode(),selected_checkpoint=v['checkpoint'],
 capacity=dict(path=str(parent),owner_uid=parent.stat().st_uid,filesystem=fs,free_bytes=shutil.disk_usage(parent).free,required_bytes=required),
 actual_gpu_contexts=0,actual_model_loads=0,actual_exports=0,semantic_quality_accepted=False)
with (r/'receipts/ldm100k_prefix_export_activation.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('SPEC', repr(s))
ast.parse(code, feature_version=(3, 8))
result = json.loads(recorded_remote(code, O / 'prefix_activate_client.json', timeout=120))
raw = base64.b64decode(result['new_spec_base64'])
assert hashlib.sha256(raw).hexdigest() == result['new_spec_sha256']
with (O / 'prefix_export_spec_before_enable.json').open('xb') as handle:
    handle.write(old)
sp.write_bytes(raw)
(O / 'prefix_export_activation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
