import fcntl,hashlib,json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';Q=R/'stage3_skill_sequence_20261103';N=R/'stage3_supermimic29_20261103/native_universal_20261104'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.pilotsummary.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
assert read(B/'pilot_postopt_v4_guard.json')['status']=='generated_pilot_postoptimization_completed_native_scene_pending'
assert read(B/'pilot_postopt_v4/independent_state_audit.json')['kinematic_passed']==3
old=next(r for r in rows(B/'pilot_postopt_v2/records.jsonl') if r['candidate_id']==101000)
latest=[old]+rows(B/'pilot_postopt_v4/records.jsonl');assert len(latest)==6 and len({r['candidate_id'] for r in latest})==6
valid=[r for r in latest if r['passed_requested_kinematic_checks']];assert len(valid)==4
counts={t:sum(r['task']==t for r in valid) for t in ['pour','bimanual_box_lift','box_push_pull']}
staged={'status':'four_kinematic_pilots_native_admission_pending','not_final_reference_library':True,'formal_training_allowed':False,'reference_fps':20,'body_dof':29,'frames':180,
 'kinematic_passed_counts':counts,'final_reference_counts':{t:0 for t in counts},'generated_candidate_counts':{t:2 for t in counts},
 'staged_candidates':valid,'remaining_failures':[r for r in latest if not r['passed_requested_kinematic_checks']],
 'audits':[str(B/'pilot_postopt_v2/independent_state_audit.json'),str(B/'pilot_postopt_v4/independent_state_audit.json')],
 'next':'native hand/object scene applicability for four staged motions; diagnose two push-pull wrist/stance failures without loosening gates or re-solving passed motions'}
atomic(B/'staged_kinematic_pilots.json',staged)
p=B/'plan.json';j=read(p);j.update(status=staged['status'],kinematic_passed_counts=counts,staged_candidates_manifest=str(B/'staged_kinematic_pilots.json'),postoptimizer_current_status='completed_normally_do_not_restart',
 final_reference_admitted_total=0,next=[staged['next'],'validated pilots then scale500 final distinct references each; no premature1500 bulk admission'])
atomic(p,j);atomic(B/'active_pipeline.json',{'status':staged['status'],'plan':str(p),'staged_candidates':str(B/'staged_kinematic_pilots.json'),'active_source_jobs':[],
 'completed_guard':str(B/'pilot_postopt_v4_guard.json'),'final_admitted_count':0,'formal_training_started':False})
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';j=read(p);j['next_action']=staged['next']
 for t in j['tasks']:
  if t['id'] in counts:t.update(status='kinematic_pilots_native_scene_pending' if counts[t['id']] else 'pilot_wrist_stance_adaptation_needed',kinematic_pilot_candidates=counts[t['id']],generated_pilot_candidates=2,admitted_references=0)
 atomic(p,j)
p=N/'all_four_reference_gate.json';j=read(p);j.update(kinematic_pilot_counts=counts,remaining_reference_generation_pipeline=str(B/'active_pipeline.json'));assert not j['training_allowed'];atomic(p,j)
print(json.dumps({'status':staged['status'],'generated_candidates':6,'kinematic_passed_counts':counts,'final_admitted_new_references':0}))
