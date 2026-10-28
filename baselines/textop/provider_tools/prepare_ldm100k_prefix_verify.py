"""New disabled independent verifier deployment; no existing source/spec edit."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sp = B / 'configs/ldm100k-prefix-verify-spec.json'
assert not sp.exists()
source = json.loads((B / 'configs/ldm100k-prefix-export-spec.json').read_bytes())
assert not source['export_authorized'] and source['ldm_training'] is None
names = ['scripts/ldm100k_export_acceptance.py', 'scripts/verify_ldm100k_prefix_export.py',
         'scripts/supervise_ldm100k_prefix_verify.py', 'provider_tools/test_ldm100k_export_acceptance_metadata.py']
for n in names:
    ast.parse((B / n).read_bytes(), filename=n, feature_version=(3, 8))
bound = set(source['bound_sources']) | {'scripts/export_ldm100k_prefix.py',
    'scripts/supervise_ldm100k_prefix_export.py', 'scripts/ldm100k_export_acceptance.py'}
s = dict(schema='textop.ldm100k_prefix_cpu_verification.v1', verify_authorized=False,
         export_binding=None, executor_sha256=sha(B / names[1]), supervisor_sha256=sha(B / names[2]),
         bound_sources={n: sha(B / n) for n in sorted(bound)}, minimum_ram_free_bytes=16 * 1024**3,
         minimum_disk_free_bytes=3 * 1024**3, reserve_bytes=2 * 1024**3,
         timeout_seconds=900, actual_verification_accepted=False,
         semantic_quality_accepted=False, training_optimizer_updates_authorized=0,
         new_gpu_contexts_authorized=0, verify_run='user_ldm100k_prefix_document_v1')
raw = (json.dumps(s, indent=2) + '\n').encode()
sp.write_bytes(raw)
payload = {n: base64.b64encode((B / n).read_bytes()).decode() for n in names}
payload['configs/ldm100k-prefix-verify-spec.json'] = base64.b64encode(raw).decode()
code = r'''
import ast,base64,hashlib,json,os,pathlib,shutil,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();items=PAYLOAD;s=POLICY
assert shutil.disk_usage(r).free>2*1024**3
for n,h in s['bound_sources'].items():
 if n not in items:assert sha(r/n)==h,n
for n,b in items.items():
 p=r/n;assert not p.exists() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 if n.endswith('.py'):ast.parse(base64.b64decode(b),filename=n)
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
out=r/'ldm100k_user_handoff_20261028/verify_metadata_server_v1';assert not out.exists()
child=subprocess.Popen([sys.executable,str(r/'provider_tools/test_ldm100k_export_acceptance_metadata.py'),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stdout,stderr=child.communicate(timeout=60);assert child.returncode==0,stderr.decode()
fixture=json.loads((out/'SUMMARY.json').read_bytes());assert fixture['rejected_cases']==18
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert not s['verify_authorized'] and s['export_binding'] is None
print(json.dumps(dict(status='LDM100K_INDEPENDENT_CPU_VERIFIER_DEPLOYED_DISABLED_NOT_ACCEPTED',
 files_sha256={n:sha(r/n) for n in items},fixture=fixture,actual_fixture_exit_code=child.returncode,
 actual_fixture_reaped=True,actual_fixture_streams_final=True,actual_fixture_stderr=stderr.decode(),
 actual_original_source_modifications=0,actual_export_source_modifications=0,actual_model_loads=0,
 actual_gpu_contexts=0,actual_production_checkpoint_reads=0,actual_verifications=0)))
'''.replace('PAYLOAD', repr(payload)).replace('POLICY', repr(s))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'verify_prepare_client.json', timeout=90))
(O / 'prefix_verify_preparation_v1.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
