import datetime,fcntl,hashlib,json,math,pathlib,statistics
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';Q=R/'stage3_skill_sequence_20261103';N=R/'stage3_supermimic29_20261103/native_universal_20261104'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.production.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
state=read(W/'controller_status.json');assert state['status']!='failed_preserved'
sampling=[]
for p in list((B/'pilot_v2').glob('*/batch_*/report.json'))+list((W/'cohorts/pour_0001/candidates').glob('batch_*/report.json')):
 j=read(p);sampling.append(j['elapsed_s']/len(j['candidates']))
post=[]
for p in [W/'optimization_witness/postopt/records.jsonl',W/'cohorts/pour_0001/postopt/records.jsonl']:
 if p.exists():post += [json.loads(x)['elapsed_seconds'] for x in p.read_text().splitlines()[1:]]
admitted=sum(state['selected_reference_counts'].values());remaining=1500-admitted
estimate={'status':'provisional_workload_estimate_not_deadline','estimated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'remaining_new_reference_target':remaining,'scene_admitted_snapshot':state['native_admitted_counts'],'sampling_seconds_per_candidate_measured_median':statistics.median(sampling),
 'postoptimization_seconds_per_candidate_measured_warm_median':statistics.median(post),'postoptimization_sample_count':len(post),
 'pure1500_candidate_sampling_hours':1500*statistics.median(sampling)/3600,
 'remaining_wall_time_planning_range_hours':[10,16],
 'range_assumptions':['continued present shared GPU/CPU availability','future native admission rate70-90 percent is a sensitivity assumption, not yet measured for all tasks','32-candidate serial generation/fitting/native cohorts','0.5-1hour final selection, mixed2000 conversion/loading and independent integrity checks'],
 'sensitivity':[{ 'assumed_admission_rate':p,'required_candidates_for_remaining':math.ceil(remaining/p),'sampling_hours_only':math.ceil(remaining/p)*statistics.median(sampling)/3600} for p in [.7,.8,.9]],
 'not_included':'SuperMimic/PPO training and final task-success evaluation; estimates will be revised from actual cohort timings/pass rates',
 'resource_pause_risk':'data floor preserved; controller waits before a new cohort if space becomes insufficient; completed logs archived losslessly'}
atomic(W/'preprocessing_eta.json',estimate)
pipeline={'status':'rolling32_reference_production_running','controller_status':str(W/'controller_status.json'),'controller_pid':state['pid'],'source_binding':str(W/'source_binding.json'),
 'scene_admitted_counts':state['native_admitted_counts'],'source_candidates_snapshot':state['sampled_new_candidates'],'first32_full_audit_passed':bool(state['cohorts_completed']),
 'ETA':str(W/'preprocessing_eta.json'),'final1500_dataset_ready':False,'formal_SuperMimic_training_started':False,'Pick_PPO_unchanged_pid':1436632}
atomic(B/'active_pipeline.json',pipeline);p=B/'plan.json';j=read(p);j.update(status=pipeline['status'],bulk_controller=str(W/'controller_status.json'),controller_pid=state['pid'],
 postoptimizer_current_status='pilot_repairs_completed_and_native_verified',all_six_pilots_native_passed=True,first32_full_audit_passed=pipeline['first32_full_audit_passed'],
 scene_admitted_counts_snapshot=state['native_admitted_counts'],ETA=str(W/'preprocessing_eta.json'));atomic(p,j)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';j=read(p);j['remaining_reference_generation_pipeline']=str(B/'active_pipeline.json');j['next_action']='Monitor immutable rolling32 bulk controller to500/task; independent mixed2000 selection/loading after1500 scene references, then native SuperMimic resource gate.'
 for t in j['tasks']:
  if t['id'] in state['native_admitted_counts']:t.update(status='rolling32_generated_reference_production',scene_admitted_references_snapshot=state['native_admitted_counts'][t['id']],final_library_count=0,live_progress=str(W/'controller_status.json'))
 atomic(p,j)
p=N/'all_four_reference_gate.json';j=read(p);j.update(remaining_reference_generation_pipeline=str(B/'active_pipeline.json'),scene_admitted_counts_snapshot=state['native_admitted_counts'],all_six_new_task_pilots_native_passed=True);assert not j['training_allowed'];atomic(p,j)
print(json.dumps({'pipeline':pipeline,'eta':estimate},indent=2))
