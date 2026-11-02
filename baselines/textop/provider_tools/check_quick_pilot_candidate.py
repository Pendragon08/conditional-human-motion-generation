"""Read-only documented witness of the DISABLED quick-pilot candidate."""
import argparse, ast, hashlib, json, pathlib
from session import remote
BASE=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
files=['scripts/quick_pilot_protocol.py','scripts/quick_pilot_trace.py','scripts/evaluate_quick_pilot.py',
       'scripts/supervise_quick_pilot.py','configs/quick-pilot-spec.json','provider_tools/validate_quick_pilot_cpu.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
local={f:sha(BASE/f) for f in files}
for f in files:
 if f.endswith('.py'):ast.parse((BASE/f).read_text(encoding='utf-8'))
spec=json.loads((BASE/'configs/quick-pilot-spec.json').read_text(encoding='utf-8'))
assert spec['pilot_launch_enabled'] is spec['sample_launch_enabled'] is False
script='''import ast,hashlib,json,pathlib,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
expected=EXPECTED
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for f,value in expected.items():
 assert sha(root/f)==value;checks.append('candidate_byte_match:'+f)
 if f.endswith('.py'):ast.parse((root/f).read_text())
spec=json.loads((root/'configs/quick-pilot-spec.json').read_text())
assert spec['pilot_launch_enabled'] is spec['sample_launch_enabled'] is False;checks.append('quick_pilot_both_gates_disabled')
qp=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text())
assert sha(root/'configs/quick-reproduction-plan-20261026.json')==spec['quick_plan_sha256'];checks.append('same_live_quick_budget_plan')
for f,value in qp['frozen_runtime_files'].items():assert sha(root/f)==value
checks.append('all16_original_training_files_unchanged')
for f,value in qp['control_sources'].items():assert sha(root/f)==value
checks.append('live_quick_control_sources_unchanged')
proof=json.loads((root/'receipts/quick_pilot_candidate_stage_v1.json').read_text())
assert proof['cpu_exit_code']==0 and proof['cpu_summary']['check_count']==28 and proof['cpu_summary']['synthetic_fixture_only'] is True
checks.append('actual_server28_new_cpu_fixtures_passed')
v2=json.loads((root/'receipts/quick_pilot_candidate_trace_v2.json').read_text())
assert v2['deployed_sha256']==expected and v2['cpu_exit_code']==0 and v2['cpu_summary']['check_count']==31
assert v2['cpu_summary']['source_sha256']==sha(root/'provider_tools/validate_quick_pilot_cpu.py')
checks.append('updated_trace_semantics31_server_checks_and_current_source_bindings_passed')
assert proof['disabled_supervisor_exit_code']!=0 and proof['disabled_no_run_directory'] is True
assert 'Quick pilot disabled' in proof['disabled_supervisor_stderr'];checks.append('actual_disabled_supervisor_refused_before_SDK')
full=json.loads((root/'configs/contact-filter-v4-spec.json').read_text())
assert all(full[k] is False for k in ['formal_evaluation_enabled','validation_launch_enabled','real_sdk_validation_completed'])
checks.append('full40k_gates_unchanged_and_disabled')
cmd=root/'upstream/TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py'
tree=ast.parse(cmd.read_text());motion=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionCommand')
names={n.name for n in motion.body if isinstance(n,ast.FunctionDef)}
assert {'joint_pos','robot_joint_pos','robot_body_pos_w','_resample_command','_update_command'}<=names
checks.append('pinned_motion_API_contains_trace_properties')
env=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py')
source=env.read_text();step=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='step')
lines=source.splitlines()[step.lineno-1:step.end_lineno]
order={key:next(i+step.lineno for i,line in enumerate(lines) if token in line) for key,token in [('termination','termination_manager.compute()'),('reward','reward_manager.compute('),('reset','self._reset_idx(reset_env_ids)'),('command','command_manager.compute(')]}
assert order['termination']<order['reward']<order['reset']<order['command'];checks.append('actual_step_terminates_then_rewards_then_autoresets_then_advances_reference')
r=json.loads((root/'tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json').read_text())
print(json.dumps(dict(status='QUICK_PILOT_DISABLED_DOCUMENT_PREFLIGHT_PASSED',check_count=len(checks),checks=checks,
 deployed_sha256=expected,pinned_motion_source_sha256=sha(cmd),pinned_step_source_sha256=sha(env),step_order_lines=order,
 observed_training_status=r['status'],observed_training_completed=r['completed_learning_iterations'],
 sdk_runs=0,training_signals_sent=0,new_optimizer_updates=0,data_deleted=False,
 real_sdk_pilot_validation_completed=False,sample_unix=time.time())))
'''.replace('EXPECTED',repr(local))
try:
 result=json.loads(remote(script,timeout=60))
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
except BaseException as error:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise
