import fcntl,json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';Q=R/'stage3_skill_sequence_20261103'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.v3record.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
g=read(B/'pilot_postopt_v3_guard.json');assert g['status'] in ['generated_pilot_postoptimization_running','generated_pilot_postoptimization_completed_native_scene_pending']
rows=[json.loads(x) for x in (B/'pilot_postopt_v2/records.jsonl').read_text().splitlines()];preserved=[r for r in rows if r['passed_requested_kinematic_checks']];assert [r['candidate_id'] for r in preserved]==[101000]
p=B/'plan.json';j=read(p);j.update(status='six_pilots_generated_audited_one_kinematic_pass_five_limit_retries_v3',
 postoptimizer_guard=str(B/'pilot_postopt_v3_guard.json'),postoptimizer_guard_pid=g['pid'],postoptimizer_python_pid=g['child_pid'],postoptimizer_current_status=g['status'],
 postoptimizer_source=str(B/'src/stage3_supported_task_postopt_v3.py'),postoptimizer_source_sha256=g['source_sha256'],
 generated_candidate_counts={'pour':2,'bimanual_box_lift':2,'box_push_pull':2},preserved_kinematic_passed_ids=[101000],
 previous_postopt_cpu_audit=str(B/'pilot_postopt_v2/independent_state_audit.json'),
 retry_ids=[100000,100001,101001,102000,102001],retry_change='only soft joint-limit penalty1000->10000; exclusive1mrad numerical correction rule and all target thresholds unchanged',
 final_reference_admitted_total=0,previous_startup_failures_preserved=True)
atomic(p,j);atomic(B/'active_pipeline.json',{'status':j['status'],'plan':str(p),'guard':j['postoptimizer_guard'],'guard_pid':g['pid'],'python_pid':g['child_pid'],'final_admitted_count':0,'formal_training_started':False})
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';q=read(p);q['remaining_reference_generation_pipeline']=str(B/'active_pipeline.json');q['next_action']='Preserve audited bimanual101000; finish strict-limit fitting retries for other5 pilots, independent state checks and task-specific native applicability, then scale each task500.';atomic(p,q)
print(json.dumps({'status':j['status'],'guard':g['status'],'pid':g['pid'],'python_pid':g['child_pid']}))
