"""Seal audited readiness and the formal resource/source contract, never launch."""
import hashlib,json,py_compile
from datetime import datetime,timezone
from r3_world_ssh_transport import ROOT,REMOTE,sync,execute
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=ROOT/'results/world_residual_ppo_20261007/raw'
runtime=json.loads((base/'runtime_doc_witness.json').read_text(encoding='utf-8'))
capacity=json.loads((base/'capacity/fresh_doc_witness_receipt_20261007.json').read_text(encoding='utf-8'))
saved=json.loads((base/'capacity/fresh_saved_tensor_audit.json').read_text(encoding='utf-8'))
evaluation=json.loads((base/'eval_interface_v3/eval_doc_witness.json').read_text(encoding='utf-8'))
assert runtime['status']=='COMPLETE_WITH_INDEPENDENT_EVIDENCE_LIMITS'
assert all(v['child_exit']==0 for v in runtime['arm_completion'].values())
assert saved['checks_all_pass'] and len(saved['checks'])==43 and all(saved['checks'].values())
assert capacity['launcher_invocations']==1 and capacity['launcher_exit']==0
assert capacity['all_independent_checks_pass'] and capacity['saved_audit_child_exit']==0
assert capacity['formal_started'] is False and capacity['formal_updates']==0
assert all(v==0 for v in capacity['scientific_child_exit'].values())
# The parent seals only the real independent trace audit after inspecting receipt.
assert evaluation['status']=='PASS_WITH_EVIDENCE_LIMITS'
assert evaluation['all_trace_checks_passed'] and evaluation['all_provenance_checks_passed']
assert evaluation['all_local_source_locks_passed']
assert evaluation['actual_child_exit']=={'P-history':0,'W-correct':0}
assert evaluation['formal_training_launched'] is False
assert evaluation['scientific_score_gate_decision_made'] is False
assert all(evaluation['cross_arm_initial_conditions_exactly_equal'].values())
assert evaluation['launcher_local_returncode']==0
for arm,receipt in evaluation['arms'].items():
 assert receipt['status']=='PASS_WITH_EVIDENCE_LIMITS' and receipt['all_trace_checks_passed']
 assert receipt['actual_child_exit']==0 and receipt['completion_sentinel_count']==1
 assert len(receipt['checks'])==33 and all(receipt['checks'].values())
 assert len(receipt['independent_trace_checks'])==19
 assert all(v=='PASS' for v in receipt['independent_trace_checks'].values())
 assert receipt['actual_evaluation_transitions']==7680
 assert receipt['max_per_reference_summary_discrepancy']==0
 assert receipt['max_macro_summary_discrepancy']==0
 assert len(receipt['evidence_limits'])==7
assert evaluation['launcher_invocation_count']==1
assert evaluation.get('PPO_updates',0)==0 and evaluation.get('aux_updates',0)==0
now=datetime.now(timezone.utc).isoformat()
checks={
 'runtime':{'ready_for_formal':True,'formal_PPO_launched':False,'at_creation_utc':now,
  'scope':'Five-arm diagnostic invariants with saved CUDA fixture supplementation; not control benefit',
  'receipt':'results/world_residual_ppo_20261007/raw/runtime_doc_witness.json',
  'receipt_sha256':sha(base/'runtime_doc_witness.json'),
  'supplemental_fixture_audit_sha256':sha(base/'capacity/fresh_saved_tensor_audit.json'),
  'v1_evidence_limits_preserved':True,'formal_v3_boundary_snapshots_required':True,
  'not_proven':['Per-minibatch gradient traces','Natural timeout semantics examples in 2-update diagnostics','500-update completion']},
 'capacity':{'ready_for_formal':True,'formal_PPO_launched':False,'at_creation_utc':now,
  'receipt':'results/world_residual_ppo_20261007/raw/capacity/fresh_doc_witness_receipt_20261007.json',
  'receipt_sha256':sha(base/'capacity/fresh_doc_witness_receipt_20261007.json'),
  'saved_tensor_audit_sha256':sha(base/'capacity/fresh_saved_tensor_audit.json'),
  'checks_passed':43,'real_transitions':786432,'natural_timeouts':0,
  'not_control_gate':True,'Torch_peak_is_not_total_driver_memory':True},
 'eval_interface':{'ready_for_formal':True,'formal_PPO_launched':False,'at_creation_utc':now,
  'receipt':'results/world_residual_ppo_20261007/raw/eval_interface_v3/eval_doc_witness.json',
  'receipt_sha256':sha(base/'eval_interface_v3/eval_doc_witness.json'),
  'source_revision':'v3_manifest_to_library_filename_mapping','native_trace_is_not_video':True,
  'scope':'64x120 frozen capacity interface, not final scientific control score',
  'PPO_updates':0,'aux_updates':0,'previous_v1_v2_failures_preserved':True,
  'independent_trace_checks_per_arm':19,'real_transitions':15360,
  'evidence_limits':{arm:receipt['evidence_limits'] for arm,receipt in evaluation['arms'].items()},
  'runtime_endpoint_assertion_not_independent_digest_proof':True}
}
out=base/'readiness';out.mkdir(parents=True,exist_ok=True)
paths=[]
for name,receipt in checks.items():
 p=out/f'{name}_independent_ready.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n')
 artifact=ROOT/f'artifacts/monitoring/r3_world_ppo_{name}_independent_ready_20261007.json'
 assert not artifact.exists();artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_bytes(p.read_bytes())
 paths +=[p.relative_to(ROOT).as_posix(),artifact.relative_to(ROOT).as_posix()]
locked=['scripts/r3_world_ppo_control_formal_v3_20261007.py','scripts/r3_world_ppo_eval_isaac_v3_20261007.py',
 'scripts/r3_world_ppo_formal_stage_v3_20261007.sh','scripts/r3_world_ppo_formal_queue_20261007.py',
 'scripts/r3_world_ppo_gate_20261007.py','scripts/launch_world_ppo_formal_20261007.py',
 'scripts/read_world_ppo_formal_status_20261007.py','scripts/r3_world_ppo_core.py',
 'scripts/r3_world_ppo_residual_core.py','scripts/r3_world_ppo_training_core_20261007.py',
 'plans/world_residual_ppo_control_metrics_20261007.md']
for name in locked:
 if name.endswith('.py'):py_compile.compile(str(ROOT/name),doraise=True)
hashes={name:sha(ROOT/name) for name in locked}
(ROOT/'configs/world_ppo_formal_sources_20261007.json').write_text(json.dumps(hashes,indent=2)+'\n')
spec={'base_spec_canonical_sha256':'47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4',
 'dependencies_unchanged':True,'resource_change_only':True,'GPU':0,'independent_renderer_GPU':1,
 'shared_idle_context_allowance_MiB':1500,'max_stage_training_s':86400,'max_stage_eval_s':1800,
 'tmux':'r3_world_ppo_formal_20261007','seeds':[42,43,44],
 'arms':['P0','P-cap','P-history','W-correct','W-shuffled-original'],
 'training':{'warm_start_updates':2000,'reference_count':66,'num_envs':8192,'rollout_steps':24,
  'additional_PPO_updates':500,'W_aux_updates_per_PPO_cycle':1,'aux_batch':1024,
  'real_transitions_per_arm_seed':98304000,'real_transitions_total':1474560000,
  'imagined_transitions':0,'PPO_reward_PD_termination_unchanged':True},
 'evaluation':{'num_envs':2048,'steps':500,'reference_count':11,'initial_phase':0,
  'reference_manifest_sha256':'9e9da9f517a7b7d8b202fd5259c740df613e6c3174d71c1f5ffa193562f17f55',
  'manifest_to_library_filename_mapping':True,'corruption':False,'actor':'deterministic',
  'PPO_aux_norm_updates':0,'transitions_total':15360000,'checkpoint_selection':'final_update500_only',
  'first_episode_failure_penalty_rad':2,'push_force_world_Y_N':80,'push_steps':[50,59],
  'recovery_steps_inclusive':[60,109],'trace_is_not_video':True},
 'explanation_arms_auto_launch':False,'MuJoCo_auto_launch':False,'innovation_validated':False,
 'source_hashes':hashes,'readiness_receipts':checks}
(ROOT/'configs/env-world-ppo-formal-20261007.json').write_text(json.dumps(spec,indent=2)+'\n')
with (ROOT/'docs/dreamcontrol_ssh_environment_ledger.md').open('a',encoding='utf-8') as f:
 f.write('\n## 2026-10-07 formal control-ablation launch contract\n\nWarm dependencies unchanged. Runtime five arms and saved43 tensor/8192 capacity checks completed; original process-proof limitations retained. Frozen evaluator v1 failed before any control trace with its error obscured by teardown; v2 exposed directoryglob order differing from manifestorder. Failures preserved. v3 maps fixed manifestID by filename to unchanged MotionLib index. Fresh independent audit is PASS_WITH_EVIDENCE_LIMITS: 19 trace checks per arm and 11-row/macro reconstructions passed, while full force/success-event/per-frame module-hash reconstruction and natural timeout/root failure examples remain limited. configs/env-world-ppo-formal-20261007.json locks fivearms×three seeds,8192×24×500 warm-start PPO,15 real frozen2048×500 evaluations. No imagined reward, extraIK, deployment or reward/PD changes. Source SHA lock configs/world_ppo_formal_sources_20261007.json. Formal readiness is not control benefit; actual first launch still requires fresh agent-follows-doc invocation docs/world_ppo_formal_invocation_20261007.md.\n')
paths += locked+['configs/world_ppo_formal_sources_20261007.json','configs/env-world-ppo-formal-20261007.json',
 'docs/world_ppo_formal_invocation_20261007.md','docs/dreamcontrol_ssh_environment_ledger.md']
print(sync(paths))
# Verify the actual old stages exited and ready records survived transport.
program=r'''
from pathlib import Path
import hashlib,json,subprocess
r=Path(REMOTE)
assert not (r/'artifacts/monitoring/r3_world_ppo_formal_20261007.state').exists()
state=(r/'artifacts/monitoring/r3_world_ppo_eval_interface_v3_20261007.state').read_text()
assert state.startswith('COMPLETE') and 'P-history_exit=0' in state and 'W-correct_exit=0' in state
for filename,expected in SOURCES.items():assert hashlib.sha256((r/filename).read_bytes()).hexdigest()==expected,filename
for name in ['runtime','capacity','eval_interface']:
 data=json.loads((r/f'artifacts/monitoring/r3_world_ppo_{name}_independent_ready_20261007.json').read_text())
 assert data['ready_for_formal'] and data['formal_PPO_launched']==False
subprocess.run(['bash','-n',str(r/'scripts/r3_world_ppo_formal_stage_v3_20261007.sh')],check=True)
print('FORMAL_CONTRACT_STAGED_NOT_LAUNCHED')
'''.replace('REMOTE',repr(REMOTE)).replace('SOURCES',repr(hashes))
print(execute(program))
