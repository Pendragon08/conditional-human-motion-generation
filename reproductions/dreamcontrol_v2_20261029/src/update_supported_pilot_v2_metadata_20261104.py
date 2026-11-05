import fcntl,json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';Q=R/'stage3_skill_sequence_20261103';N=R/'stage3_supermimic29_20261103/native_universal_20261104'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.pilotupdate.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
guard=read(B/'pilot_postopt_v2_guard.json');assert guard['status']=='generated_pilot_postoptimization_running'
p=B/'plan.json';j=read(p);j.update(postoptimizer_guard=str(B/'pilot_postopt_v2_guard.json'),postoptimizer_guard_pid=guard['pid'],postoptimizer_python_pid=guard['child_pid'],
 postoptimizer_source=str(B/'src/stage3_supported_task_postopt_v2.py'),postoptimizer_source_sha256=guard['source_sha256'],
 previous_postoptimizer_v1_failure='180frame temporal extension left119 smoothness pairs; v2 uses179, no threshold changes; old log/source/output preserved',
 previous_sampler_v1_failure='visibility environment rejected by existing setup before any denoising; fixed queue_v2 preserves physical CUDA indices, no environment changes',
 current_generated_candidate_counts={'pour':2,'bimanual_box_lift':2,'box_push_pull':2},source_candidate_audit_passed=True)
atomic(p,j)
atomic(B/'active_pipeline.json',{'status':'generated_six_pilot_candidates_audited_postoptimization_running','plan':str(p),'guard':j['postoptimizer_guard'],'guard_pid':guard['pid'],'python_pid':guard['child_pid'],'final_admitted_count':0,'formal_training_started':False})
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';j=read(p);j['remaining_reference_generation_pipeline']=str(B/'active_pipeline.json');atomic(p,j)
print(json.dumps({'status':'actual_pipeline_metadata_updated','guard_pid':guard['pid'],'python_pid':guard['child_pid']}))
