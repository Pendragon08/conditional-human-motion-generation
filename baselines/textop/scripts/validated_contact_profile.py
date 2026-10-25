"""Adopt an independently exercised contact representation; no training math changes."""
import hashlib,json,pathlib

def sha(path):
 return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def apply(cfg,output,root,profile):
 root=pathlib.Path(root);output=pathlib.Path(output)
 assert profile['kind']=='validated_static_solid_floor' and profile['enabled'] is True
 assert profile['training_math_changes'] is False and profile['self_collisions'] is True
 for name,expected in profile['bound_files'].items():
  assert sha(root/name)==expected,'Validated source/profile changed: '+name
 assert len(profile['witnesses'])==2 and len({x['run'] for x in profile['witnesses']})==2
 proofs=[]
 for item in profile['witnesses']:
  run=item['run'];out=root/'tracker_validation'/run
  receipt=out/'receipt.json';exitpath=root/'receipts'/('full_contact_'+run+'_exit.json')
  assert sha(receipt)==item['receipt_sha256'] and sha(exitpath)==item['exit_sha256']
  r=json.loads(receipt.read_text());e=json.loads(exitpath.read_text())
  assert e['child_exit_code']==0 and e['timed_out'] is False
  assert r['status']=='TEXTOP_SOLID_FLOOR_PPO_VALIDATION_COMPLETED' and r['output_writers_closed']
  assert r['completed_learning_iterations']==10 and r['optimizer_updates']==200 and r['new_checkpoints_written']==0
  assert r['reference_sources']==8278 and r['num_envs']==8192 and r['environment_transitions']==1966080
  assert r['numerical_alert_free'] is True and r['parameter_max_change']>0
  audit=r['native_state_audit']
  for key in ['cached_vs_original_root','cached_vs_original_joint_position','cached_vs_original_joint_velocity','native_vs_cached_joint_position','native_vs_cached_joint_velocity','selected_first_action_difference']:
   assert audit[key]['abs_max']==0
  t=r['numerical_trace'];assert t['physics_samples']==960 and t['reward_samples']==240
  assert t['first_state_alert'] is None and t['first_reward_alert'] is None
  trace=out/'numerical_trace.jsonl';assert sha(trace)==t['trace_sha256']==item['trace_sha256']
  for line in trace.read_text().splitlines():
   row=json.loads(line)
   for value in list(row.get('fields',{}).values())+[row.get('reward',{}),row.get('contact_force_components',{})]+list(row.get('weighted_reward_rates',{}).values()):
    assert value.get('nonfinite',0)==0
  assert r['source_sha256']==profile['bound_files']['scripts/solid_floor_ppo_probe.py']
  assert r['paper_config_sha256']==profile['bound_files']['scripts/paper_filter_config.py']
  assert r['recording_runtime_sha256']==profile['bound_files']['scripts/filter_training_runtime.py']
  assert r['capacity_spec_sha256']==profile['bound_files']['configs/full-library-capacity-spec.json']
  assert r['inputs']['reference_order_fingerprint']==profile['reference_order_fingerprint']
  assert r['backup_audit_sha256']==profile['backup_audit_sha256']
  proofs.append({'run':run,'receipt_sha256':item['receipt_sha256'],'actual_iterations':10,'actual_optimizer_updates':200})
 assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
 assert cfg.sim.dt==0.005 and cfg.decimation==4
 from solid_floor_diagnostic import create
 cfg.scene.terrain.usd_path=create(output/'solid_floor.usda')
 return {'kind':profile['kind'],'top_z':0.0,'dimensions_m':[4000,4000,0.2],
         'static_friction':1.0,'dynamic_friction':1.0,'restitution':0.0,'combine_mode':'multiply',
         'self_collisions':True,'validation_witnesses':proofs,
         'author_exact_collision_geometry_available':False,
         'long_run_convergence_established':False,'data_or_reward_changes':False}
