"""Reconcile the closed pre-child role rejection and centralize the new role IDs."""
import ast
import base64
import hashlib
import json
import pathlib
import sys
import types
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
old_sources = json.loads((B / 'sources/ldm100k_generation_failed_author_v2/source_bytes.json').read_bytes())
sp = B / 'configs/ldm100k-generation-spec.json'
old = sp.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'e319ac3e62f6f7d0cda90c69afb567cdfa7bc84531116d0031bcfc88fcd4ec7f'
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_generation_acceptance import RUNS

names = ['scripts/ldm100k_generation_acceptance.py', 'scripts/run_ldm100k_heldout_generation.py',
         'scripts/supervise_ldm100k_heldout_generation.py']
payload = {}
for name in names:
    raw = (B / name).read_bytes()
    ast.parse(raw.decode(), feature_version=(3, 8))
    payload[name] = dict(old_sha256=old_sources[name]['sha256'],
                         new_sha256=hashlib.sha256(raw).hexdigest(),
                         new_base64=base64.b64encode(raw).decode())
# Exercise the actual three entry guards without executing an entry, model or fixture.
guard_checks = 0
for name in names[1:] + ['provider_tools/launch_ldm100k_generation.py']:
    tree = ast.parse((B / name).read_text(encoding='utf-8'))
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)
              and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
              and isinstance(n.test.left.value, ast.Name) and n.test.left.value.id == 'a'
              and n.test.left.attr == 'run']
    assert len(guards) == 1
    guard = compile(ast.Expression(guards[0].test), name, 'eval')
    for mode, run, expected in [('witness', RUNS[0], True), ('witness', RUNS[1], True),
                                ('witness', 'user_ldm100k_generation_author_v2', False),
                                ('diagnostic', 'user_ldm100k_diagnostic32_v1', True)]:
        assert bool(eval(guard, dict(a=types.SimpleNamespace(mode=mode, run=run), RUNS=RUNS))) == expected
        guard_checks += 1

code = r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned
from ldm100k_scope_acceptance import accepted_scope
from contact_filter_resources import sample_resources
sp=r/'configs/ldm100k-generation-spec.json';assert sha(sp)==OLD_SHA;s=read(sp);payload=PAYLOAD
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
run='user_ldm100k_generation_author_v2';prefix='ldm100k_generation_'+run
outer=owned(r/'receipts'/(prefix+'_outer_exit.json'));driver=owned(r/'receipts'/(prefix+'_driver.json'));log=owned(r/'logs'/(prefix+'_outer.log'))
assert read(outer)['actual_returncode']==1 and read(driver)['spec_sha256']==sha(sp)
assert 'supervise_ldm100k_heldout_generation.py' in log.read_text() and 'AssertionError' in log.read_text()
assert not (r/'ldm100k_user_handoff_20261028'/run).exists()
for p in [r/'receipts'/(prefix+x) for x in ['_exit.json','_process_exit.json']]+[r/'logs'/(prefix+x) for x in ['.log','_resources.jsonl']]:assert not p.exists()
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=1013:continue
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['run_ldm100k_heldout_generation.py','supervise_ldm100k_heldout_generation.py']):workers.append(int(p.name))
 except (OSError,ProcessLookupError):pass
assert not workers
for n,h in s['bound_sources'].items():assert sha(owned(r/n))==h,n
assert sha(r/'scripts/run_ldm100k_heldout_generation.py')==s['executor_sha256'] and sha(r/'scripts/supervise_ldm100k_heldout_generation.py')==s['supervisor_sha256']
assert s['witnesses_authorized'] and not s['generation_authorized'] and s['witness_bindings'] is None
accepted_scope(s['scope_binding']);res=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes'])
assert res['data_free_bytes']>=s['reserve_bytes']+s['log_allowance_bytes']+s['accepted_output_upper_bytes']['witness']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
assert any(int(x.split(',')[0])==1 and int(x.split(',')[1])>=20*1024 for x in gpu.splitlines())
archive=r/'receipts/ldm100k_generation_failed_author_v2';archive.mkdir(mode=0o700,exist_ok=False)
for name in ['configs/ldm100k-generation-spec.json']+list(payload):
 src=owned(r/name);dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if name in payload:assert sha(src)==payload[name]['old_sha256']
 with dest.open('xb') as f:f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
proof=dict(outer_sha256=sha(outer),driver_sha256=sha(driver),outer_log_sha256=sha(log),actual_outer_exit=1,no_child_receipt_process_resources_or_outputs=True,workers=workers,new_model_loads=0,new_gpu_contexts=0)
for name,item in payload.items():
 raw=base64.b64decode(item['new_base64']);assert hashlib.sha256(raw).hexdigest()==item['new_sha256'];ast.parse(raw.decode());p=r/name;temp=p.with_name(p.name+'.identity-repair.partial')
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 temp.replace(p)
 if name in s['bound_sources']:s['bound_sources'][name]=item['new_sha256']
s['executor_sha256']=payload['scripts/run_ldm100k_heldout_generation.py']['new_sha256'];s['supervisor_sha256']=payload['scripts/supervise_ldm100k_heldout_generation.py']['new_sha256'];s['inference_contract_revision']=3
raw=(json.dumps(s,indent=2)+'\n').encode();temp=sp.with_name(sp.name+'.identity-repair.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(sp)
result=dict(status='PRECHILD_IDENTITY_REJECTION_RECONCILED_REVISION3_WITNESSES_ENABLED_NOT_DISPATCHED',observed_unix=time.time(),new_spec_sha256=sha(sp),new_spec_base64=base64.b64encode(raw).decode(),failure_proof=proof,changed_sources={n:x['new_sha256'] for n,x in payload.items()},preflight_resources=res,numerical_settings_or_weights_changed=False,new_model_loads=0,new_gpu_contexts=0)
with (archive/'repair.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('PAYLOAD', repr(payload))
ast.parse(code, feature_version=(3, 8))
proof = dict(actual_guard_checks=guard_checks, all_three_actual_entry_guard_expressions_tested=True,
             positive_current_roles_and_diagnostic_accepted=True, old_failed_role_rejected=True,
             new_model_loads=0, new_gpu_contexts=0, actual_numerical_acceptance=False)
with (O / 'generation_identity_v3_guard_checks.json').open('x', encoding='utf-8') as f:
    json.dump(proof, f, indent=2)
result = json.loads(recorded_remote(code, O / 'generation_identity_repair_v3_client.json', timeout=180))
raw = base64.b64decode(result['new_spec_base64'])
assert hashlib.sha256(raw).hexdigest() == result['new_spec_sha256']
sp.write_bytes(raw)
(O / 'generation_identity_repair_v3.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'new_spec_base64'}))
