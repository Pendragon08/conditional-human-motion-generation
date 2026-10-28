"""Stage a disabled heldout scope helper and its new isolated CPU fixtures.

No corpus rebuild, production array/text read, model load or generation launch.
The actual scope builder, generator and trained TMR evaluator remain unaccepted.
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
sp = B / 'configs/ldm100k-heldout-plan.json'
assert not sp.exists()
names = ['scripts/ldm100k_heldout_protocol.py', 'provider_tools/test_ldm100k_heldout_scope.py']
for n in names:
    ast.parse((B / n).read_bytes(), feature_version=(3, 8))
s = dict(schema='textop.ldm100k_heldout_plan.v1', scope_build_authorized=False,
         generation_authorized=False, model_bindings=None, actual_input_scope_accepted=False,
         helper_sha256=sha(B / names[0]), seeds=[20261028, 20261029, 20261030, 20261031, 20261101],
         modes=['conditional_cfg5', 'unconditional'], fps=50, feature_dim=69, dof=29,
         history=2, future=8, maximum_segment_future_frames=200,
         first_diagnostic_family_limit=32, first_diagnostic_is_full_validation_benchmark=False,
         full_validation_family_limit=None, diffusion_steps=5, cfg=5,
         timestamp_rounding='floor start, ceil end; target_start>=2; target_end<=T-1',
         initial_history='two past reference features; later history is generated only',
         future_posterior_conditioning_authorized=False,
         transition_benchmark_implemented=False, text_motion_feature_extractor=None,
         tmr_rprecision_mmdist_fid_accepted=False, actual_gpu_or_sdk_launches=0,
         actual_scope_builds=0, actual_generation_runs=0)
raw = (json.dumps(s, indent=2) + '\n').encode()
sp.write_bytes(raw)
payload = {n: base64.b64encode((B / n).read_bytes()).decode() for n in names}
payload['configs/ldm100k-heldout-plan.json'] = base64.b64encode(raw).decode()
frozen_names = ['configs/user-vae100k-ldm-spec.json', 'scripts/run_user_vae100k_ldm.py',
                'scripts/filtered15000_ldm_runtime.py', 'configs/ldm100k-user-control-spec.json',
                'scripts/watch_ldm100k_handoff.py', 'configs/ldm100k-prefix-export-spec.json',
                'configs/ldm100k-prefix-verify-spec.json']
frozen = {n: sha(B / n) for n in frozen_names}
code = r'''
import ast,base64,hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();items=PAYLOAD;frozen=FROZEN
for n,h in frozen.items():assert sha(r/n)==h,n
for n,b in items.items():
 p=r/n;assert not p.exists() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 if n.endswith('.py'):ast.parse(base64.b64decode(b),filename=n)
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
out=r/'ldm100k_user_handoff_20261028/heldout_scope_fixture_server_v1';assert not out.exists()
child=subprocess.Popen([sys.executable,str(r/'provider_tools/test_ldm100k_heldout_scope.py'),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stdout,stderr=child.communicate(timeout=60);assert child.returncode==0,stderr.decode()
fixture=json.loads((out/'SUMMARY.json').read_bytes());assert fixture['rejected_cases']==16
for n,h in frozen.items():assert sha(r/n)==h,n
print(json.dumps(dict(status='HELDOUT_SCOPE_HELPER_DEPLOYED_DISABLED_NOT_ACTUAL_SCOPE',
 files_sha256={n:sha(r/n) for n in items},frozen_sources_sha256=frozen,fixture=fixture,
 actual_fixture_exit_code=child.returncode,actual_fixture_reaped=True,actual_fixture_streams_final=True,
 actual_model_loads=0,actual_original_arrays_read=0,actual_generation_runs=0,
 actual_semantic_metrics=0,actual_production_signals=0,actual_gpu_contexts=0)))
'''.replace('PAYLOAD', repr(payload)).replace('FROZEN', repr(frozen))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'heldout_prepare_client.json', timeout=90))
(O / 'heldout_protocol_preparation_v1.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))
