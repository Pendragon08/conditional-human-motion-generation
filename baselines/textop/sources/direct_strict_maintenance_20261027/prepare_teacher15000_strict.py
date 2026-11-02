"""Create a separate frozen15k full replay without editing production sources."""
import pathlib,json,hashlib,ast,sys,base64
W=pathlib.Path(__file__).resolve().parents[2]
B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(rel,text):
 p=B/rel;assert not p.exists(),p;p.write_text(text,encoding='utf-8');ast.parse(text) if p.suffix=='.py' else None
def adapt(text):
 return text.replace('teacher10000_pilot','teacher15000_strict').replace('teacher10000-pilot-spec.json','teacher15000-direct-strict-spec.json').replace('TEACHER10000_PILOT','TEACHER15000_STRICT').replace("'sample'","'full'").replace('sample_launch_enabled','full_launch_enabled').replace('quick_pilot_trace','direct_strict_trace').replace('==10000','==15000').replace('==200000','==300000').replace('iterations=10000','iterations=15000').replace('updates=200000','updates=300000')
s=adapt((B/'scripts/evaluate_teacher10000_pilot.py').read_text(encoding='utf-8'))
s=s.replace('"""Separate10k paired pilot. No whole-library acceptance or fabricated5k/40k counts."""','"""User-directed frozen15k strict complete replay, with no training."""')
s=s.replace("'64 whole originals50..2000frames; round-robin dataset/duration buckets; stable source hash; no EMA selection'","'All8278 originals once; sorted by (duration, original index); direct strict replay without EMA selection'")
s=s.replace("env_count=spec[a.stage+'_parallel_envs'];assert env_count==(8 if a.stage=='smoke' else 16)","env_count=spec[a.stage+'_parallel_envs'];assert env_count==(8 if a.stage=='smoke' else 16)\n planned_calls=sum(max(rows[i]['frames'] for i in eligible[k:k+env_count])-1 for k in range(0,len(eligible),env_count))\n assert planned_calls<=spec[a.stage+'_max_control_calls']\n assert sum(rows[i]['frames'] for i in eligible)*36*4<=spec['maximum_private_trace_bytes']\n receipt.update(planned_maximum_control_calls=planned_calls,strict_rollouts_per_motion=1,direct_replay_bypasses_historical_ema_selection=True)")
s=s.replace("quick_pilot_executed=True,physical_tracking_executed=True","quick_pilot_executed=a.stage=='smoke',physical_tracking_executed=True")
s=s.replace("Quick replay cannot optimize","Frozen strict replay cannot optimize").replace('Quick call budget exceeded','Strict call budget exceeded')
s=s.replace("actual_scene_physics_steps=control_calls*4);save()","actual_scene_physics_steps=control_calls*4)\n    if a.stage=='smoke' or control_calls%50==0 or not pending:save()")
s=s.replace("status='QUICK_PILOT_TRIALS_FINISHED',diagnostic_only=True","status='DIRECT_STRICT_TRIALS_FINISHED',diagnostic_only=a.stage=='smoke'")
s=s.replace("receipt.update(status=STATUS,output_writers_closed=True","receipt.update(status=STATUS,output_writers_closed=True,physical_filter_executed=a.stage=='full'")
s=s.replace("'Intermediate10k teacher; limited selected whole-motion trials, not whole-library physical filtering or paper quality'","'Frozen15000, one deterministic strict complete replay per original. Direct user route bypasses historical EMA selection. Final reference endpoint and semantic/paper quality are not established. Corpus acceptance is separate.'")
assert '10000' not in s and '200000' not in s
put('scripts/evaluate_teacher15000_strict.py',s)
s=adapt((B/'scripts/supervise_teacher10000_pilot.py').read_text(encoding='utf-8'))
s=s.replace('"""A separate, initially disabled10k pilot; never stops or overlaps a trainer."""','"""Own locked frozen15k strict replay; no training or overlapping SDK worker."""')
s=s.replace("'evaluate_teacher15000_strict.py']","'evaluate_teacher10000_pilot.py','evaluate_teacher15000_strict.py']")
s=s.replace("spec['minimum_preflight_disk_free_gib']","spec[a.stage+'_minimum_preflight_disk_free_gib']")
s=s.replace("physical_filter_executed=False","physical_filter_executed=bool(a.stage=='full' and r.get('status')==STATUS)")
put('scripts/supervise_teacher15000_strict.py',s)
s=adapt((B/'provider_tools/launch_teacher10000_pilot.py').read_text(encoding='utf-8'))
s=s.replace('"""One durable, bounded10k replay; never retries an existing run."""','"""One durable frozen15k replay, with exclusive client intent and output."""')
s=s.replace('from session import remote','from backup_filter_recovery_checkpoint import recorded_remote')
s=s.replace("j=json.loads(remote(code,timeout=40));out=B/'results/teacher10000_actual_20261027'/a.run;out.mkdir(exist_ok=False)","out=B/'results/teacher15000_direct_strict_20261027'/a.run;out.mkdir(parents=True,exist_ok=False)\nj=json.loads(recorded_remote(code,out/'dispatch.json',timeout=60))")
put('provider_tools/launch_teacher15000_strict.py',s)
spec=json.loads((B/'configs/teacher10000-pilot-spec.json').read_text(encoding='utf-8'))
for k in list(spec):
 if k.startswith('sample_') or k in ['minimum_preflight_disk_free_gib','independent_summary_at_activation_sha256']:del spec[k]
spec.update(schema='textop.teacher15000_direct_strict.v1',quality_label='USER_DIRECT_STRICT_15000_NOT_PAPER_TWO_STAGE_FILTER',
 pilot_launch_enabled=True,full_launch_enabled=False,smoke_witnesses=[],full_parallel_envs=16,full_source_count=8278,
 full_max_control_calls=2000000,full_timeout_seconds=86400,checkpoint_completed=15000,checkpoint_updates=300000,
 teacher_run='filter_policy_paper69_resume10000_filter_recovery_20261027_v1',
 expected_checkpoint_sha256='66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88',
 control_spec_sha256=sha(B/'configs/teacher15000-direct-evaluation-spec.json'),
 maximum_private_trace_bytes=1073741824,smoke_minimum_preflight_disk_free_gib=2.5,full_minimum_preflight_disk_free_gib=3.5,
 full_source_order='sort(frame_count, original_global_index)',strict_rollouts_per_motion=1,
 limitation='Explicit user direct full-library strict replay; historical EMA5% candidate count remains0; not the paper two-stage selection route. One deterministic rollout/source; final endpoint and semantic quality require separate evaluation.')
for n,k in [('evaluate_teacher15000_strict.py','executor_sha256'),('teacher15000_strict_protocol.py','protocol_sha256'),('direct_strict_trace.py','trace_sha256'),('supervise_teacher15000_strict.py','supervisor_sha256')]:spec[k]=sha(B/'scripts'/n)
put('configs/teacher15000-direct-strict-spec.json',json.dumps(spec,indent=2)+'\n')
print(json.dumps({k:spec[k] for k in ['executor_sha256','protocol_sha256','trace_sha256','supervisor_sha256','full_launch_enabled']}))
