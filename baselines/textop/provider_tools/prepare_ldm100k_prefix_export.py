"""Deploy only new DISABLED CPU export sources, then isolated stdlib negatives.

This neither activates nor launches a model/export. All active LDM/controller
hashes are checked before and after, without taking the occupied GPU lock.
"""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sp = B / 'configs/ldm100k-prefix-export-spec.json'
assert not sp.exists()
policy = json.loads((B / 'configs/ldm100k-user-control-spec.json').read_bytes())
names = ['scripts/ldm100k_prefix_completion.py', 'scripts/export_ldm100k_prefix.py',
         'scripts/supervise_ldm100k_prefix_export.py', 'provider_tools/test_ldm100k_prefix_metadata.py']
for n in names:
    ast.parse((B / n).read_bytes(), filename=n, feature_version=(3, 8))
bound = set(policy['bound_sources']) | {
    'configs/ldm100k-user-control-spec.json', 'scripts/watch_ldm100k_handoff.py',
    'scripts/supervise_ldm100k_handoff.py', 'scripts/ldm100k_prefix_completion.py',
    'scripts/filtered15000_vae_completion.py'}
s = dict(schema='textop.ldm100k_prefix_cpu_export.v1', export_authorized=False,
         ldm_training=None, executor_sha256=sha(B / names[1]), supervisor_sha256=sha(B / names[2]),
         bound_sources={n: sha(B / n) for n in sorted(bound)},
         export_dir='/home/ouyanghao/textop-user-vae100k-ldm-dual-20261028/inference',
         export_allowed_root='/home/ouyanghao/textop-user-vae100k-ldm-dual-20261028',
         export_upper_bound_bytes=128 * 1024**2, reserve_bytes=2 * 1024**3,
         log_allowance_bytes=128 * 1024**2, minimum_ram_free_bytes=16 * 1024**3,
         minimum_disk_free_bytes=3 * 1024**3, timeout_seconds=900,
         actual_endpoint_accepted=False, semantic_quality_accepted=False,
         selected_user_endpoint=100000, original_native_schedule=300000,
         training_optimizer_updates_authorized=0, new_gpu_contexts_authorized=0)
raw = (json.dumps(s, indent=2, ensure_ascii=False) + '\n').encode()
sp.write_bytes(raw)
payload = {n: base64.b64encode((B / n).read_bytes()).decode() for n in names}
payload['configs/ldm100k-prefix-export-spec.json'] = base64.b64encode(raw).decode()
code = r'''
import ast,base64,hashlib,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});s=POLICY;items=PAYLOAD
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['bound_sources'].items():
 if n not in items:assert sha(r/n)==h,n
assert shutil.disk_usage(r).free>2*1024**3
for n,b in items.items():
 p=r/n;assert not p.exists() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 raw=base64.b64decode(b)
 if n.endswith('.py'):ast.parse(raw,filename=n)
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
out=r/'ldm100k_user_handoff_20261028/prefix_metadata_server_v1';assert not out.exists()
started=time.monotonic()
child=subprocess.Popen([sys.executable,str(r/'provider_tools/test_ldm100k_prefix_metadata.py'),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stdout,stderr=child.communicate(timeout=60)
assert child.returncode==0,(child.returncode,stderr.decode())
fixture=json.loads((out/'SUMMARY.json').read_bytes());assert fixture['rejected_cases']==20
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert not s['export_authorized'] and s['ldm_training'] is None
print(json.dumps(dict(status='LDM100K_PREFIX_EXPORT_SOURCES_DEPLOYED_DISABLED_NOT_ACCEPTED',files_sha256={n:sha(r/n) for n in items},
 actual_fixture=fixture,actual_fixture_exit_code=child.returncode,actual_fixture_reaped=True,actual_fixture_streams_final=True,
 actual_fixture_stdout=stdout.decode(),actual_fixture_stderr=stderr.decode(),elapsed_seconds=time.monotonic()-started,
 original_training_and_controller_source_modified=False,actual_model_loads=0,actual_gpu_contexts=0,
 actual_production_checkpoint_reads=0,actual_export_launches=0,actual_source_acceptance=False)))
'''.replace('POLICY', repr(s)).replace('PAYLOAD', repr(payload))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'prefix_prepare_client.json', timeout=90))
(O / 'prefix_export_preparation_v1.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
