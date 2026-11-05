import fcntl,json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';Q=R/'stage3_skill_sequence_20261103'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.v4record.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
g=read(B/'pilot_postopt_v4_guard.json');assert g['status'] in ['generated_pilot_postoptimization_running','generated_pilot_postoptimization_completed_native_scene_pending']
old=read(B/'pilot_postopt_v3_guard.json');assert old['status']=='generated_pilot_postoptimization_completed_native_scene_pending' and old['result']['kinematic_passed']==0
p=B/'plan.json';j=read(p);j.update(status='six_pilots_generated_audited_one_preserved_warm_limit_retries_v4',
 postoptimizer_guard=str(B/'pilot_postopt_v4_guard.json'),postoptimizer_guard_pid=g['pid'],postoptimizer_python_pid=g['child_pid'],postoptimizer_current_status=g['status'],
 postoptimizer_source=str(B/'src/stage3_supported_task_postopt_v4.py'),postoptimizer_source_sha256=g['source_sha256'],
 retry_change='same strict limits and soft weight10000; initialize from hash-bound previous near-task raw state clipped only for optimization starting values; saved raw and actual initialization hashes',
 previous_v3_result='joint limits passed but wrist/foot checks failed5/5; preserved and not admitted',final_reference_admitted_total=0)
atomic(p,j);atomic(B/'active_pipeline.json',{'status':j['status'],'plan':str(p),'guard':j['postoptimizer_guard'],'guard_pid':g['pid'],'python_pid':g['child_pid'],'final_admitted_count':0,'formal_training_started':False})
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';q=read(p);q['remaining_reference_generation_pipeline']=str(B/'active_pipeline.json');q['next_action']='Finish v4 warm initialization fitting of5 pilot failures, keep101000 unchanged, independently validate exact state/target/limits then native applicability before scale500 each.';atomic(p,q)
print(json.dumps({'status':g['status'],'pid':g['pid'],'python_pid':g['child_pid']}))
