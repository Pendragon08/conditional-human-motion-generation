import datetime,fcntl,hashlib,json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';Q=R/'stage3_skill_sequence_20261103';N=R/'stage3_supermimic29_20261103/native_universal_20261104'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.pilot.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
audit=read(B/'pilot_v2/candidate_audit.json');assert audit['candidate_count']==6
opt=read(B/'pilot_postopt_v1_launch.json')
plan={'status':'six_candidate_pilot_postoptimization_running','tasks':['pour','bimanual_box_lift','box_push_pull'],
 'final_reference_target_per_task':500,'final_reference_target_total':1500,'pilot_generated_per_task':2,'pilot_generated_total':6,'final_admitted_per_task':0,
 'source_prior_checkpoint_sha256':'109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9',
 'body_dof':29,'reference_fps':20,'pilot_frames':180,'pilot_duration_s':9.,
 'candidate_audit':str(B/'pilot_v2/candidate_audit.json'),'candidate_audit_sha256':hashlib.sha256((B/'pilot_v2/candidate_audit.json').read_bytes()).hexdigest(),
 'postoptimizer_guard':str(B/'pilot_postopt_v1_guard.json'),'postoptimizer_guard_pid':opt['pid'],
 'success_condition':'not candidate counts: actual task-specific29DoF fit, source/target/state checks, calibrated scene/native applicability and mixed loader evidence',
 'planned_object_paths_are_not_simulated_motion':True,
 'next':['finish six postoptimized pilots and independent state/target checks','task-specific native hand/object scene applicability','only then scale each task to500 final distinct references'],
 'SuperMimic_formal_gate':str(N/'all_four_reference_gate.json'),'PPO_running_unchanged_pid':1436632}
atomic(B/'plan.json',plan)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=Q/'progress.json';j=read(p);j['remaining_reference_generation_pipeline']=str(B/'plan.json');j['next_action']='Complete three-task pilots: actual six candidates generated and independently audited;29DoF dual/rotating-wrist fitting running, then native scene checks and scale each task500.'
 for t in j['tasks']:
  if t['id'] in plan['tasks']:t.update(status='six_candidate_pilot_postoptimization_running',generated_pilot_candidates=2,admitted_references=0,pipeline_plan=str(B/'plan.json'))
 atomic(p,j)
p=N/'all_four_reference_gate.json';j=read(p);j.update(candidate_counts={'pour':2,'bimanual_box_lift':2,'box_push_pull':2},remaining_reference_generation_pipeline=str(B/'plan.json'));assert not j['training_allowed'];atomic(p,j)
print(json.dumps(plan,indent=2))
