"""Complete read-only evidence on the retained failed witness; never grants PASS."""
from pathlib import Path
import collections,hashlib,json,sys
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation');OUT=ROOT/'results/support_reference_probe_v63_20261012';BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
sys.path.insert(0,str(ROOT/'scripts'))
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
r=read(OUT/'DOCUMENT_WITNESS.json');assert r['status']=='FAIL'
initial_json=OUT/'DOCUMENT_WITNESS_INITIAL_FAIL.json';initial_md=OUT/'DOCUMENT_WITNESS_INITIAL_FAIL.md'
with initial_json.open('xb') as f:f.write((OUT/'DOCUMENT_WITNESS.json').read_bytes())
with initial_md.open('xb') as f:f.write((OUT/'DOCUMENT_WITNESS.md').read_bytes())
source=BASE/'raw/val_shared_s11_r0';batch=OUT/'raw/val_baseline_s11'
sm=read(source/'metadata.json');sr=read(source/'metrics.json');sa=load(source/'trajectories.npz')
bm=read(batch/'metadata.json');br=read(batch/'metrics.json');ba=load(batch/'trajectories.npz');trace=load(batch/'observation_trace.npz');ref=load(OUT/'references/baseline.npz')['reference'];receipt=read(batch/'receipt.json')
canonical=(ref-ref[:,:,0:1])+ref[:,:,0:1];delta=ba['reference']-canonical;raw_delta=ba['reference']-ref
loc=np.unravel_index(np.abs(delta).argmax(),delta.shape)
r['failure_detail']=dict(check='applied_reference_canonical_exact',required_max_error_m=0.0,actual_max_error_m=float(np.abs(delta).max()),raw_input_max_error_m=float(np.abs(raw_delta).max()),max_error_index=list(map(int,loc)),frozen_reference_value=float(ref[loc]),canonical_reference_value=float(canonical[loc]),actual_applied_reference_value=float(ba['reference'][loc]),different_scalar_count=int(np.count_nonzero(delta)),total_scalars=int(delta.size),initial_failure_json_sha256=sha(initial_json),initial_failure_md_sha256=sha(initial_md),no_tolerance_relaxation=True,no_rerun_or_source_fix=True)
def c(n,v):r['checks'][n]=bool(v)
previous=np.concatenate([trace['ref_position'][:1],trace['ref_position'][:-1]],axis=0);vel=(trace['ref_position']-previous)/np.float32(bm['dt']);vel_error=float(np.abs(trace['ref_velocity']-vel).max())
c('observation_trace_hash',sha(batch/'observation_trace.npz')==bm['observation_trace_sha256'])
c('reference_trace_same_applied_clock',np.array_equal(trace['ref_position'],ba['reference']))
c('reference_previous_successive_exact',np.array_equal(trace['ref_previous'],previous))
c('reference_velocity_successive_exact',vel_error==0 and not trace['ref_velocity'][0].any())
c('raw_task_interface',trace['task_observation'].shape==(120,72,574) and bm['policy_observation_mode']==1 and bm['policy_observation_size']==574 and np.isfinite(trace['task_observation']).all())
c('physx_cuda_cpu_export_completion_fence',not bm['sim_params']['use_gpu_pipeline'] and bm['sim_params']['physx_use_gpu'] and bm['completion_fence_calls']==242 and bm['contact_mutation_during_reward_max_N']==0)
c('same_clock_actual_state_shape',ba['actual'].shape==ba['reference'].shape==(120,72,24,3) and np.isfinite(ba['actual'][ba['valid']]).all())
own=read(OUT/'budget_ledger.json');charged_source=[x for x in read(BASE/'budget_ledger.json') if x['label']=='val_shared_s11_r0']
c('single_charged_attempt',len(own)==1 and own[0]['label']=='val_baseline_s11' and own[0]['exit_code']==0 and own[0]['request_sha256']==sha(OUT/'requests/val_baseline_s11.json'))
c('source_build_charged_not_zero',len(charged_source)==1 and charged_source[0]['seconds']>0 and charged_source[0]['exit_code']==0)
prior=read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1]);dirs=[ROOT/n for n in prior['directories']]
if OUT not in dirs:dirs.append(OUT)
all_rows=[x for d in dirs if (d/'budget_ledger.json').exists() for x in read(d/'budget_ledger.json')]
total=sum(x['seconds'] for x in all_rows);own_seconds=sum(x['seconds'] for x in own);disk_bytes=sum(f.stat().st_size for d in dirs for f in d.rglob('*') if f.is_file())+prior['archive_bytes']
c('own_600_second_cap',own_seconds<600);c('global_72_gpu_hour_cap',total<72*3600);c('global_15gib_disk_cap',disk_bytes<15*1024**3)
r['budget']=dict(charged_baseline_ledger=own,shared_source_native_generation_and_first_build_ledger=charged_source,shared_source_generation_batch_calls=sm['generation_batch_calls'],shared_source_cost_reused_not_zero=True,own_charged_seconds=own_seconds,own_remaining_seconds=600-own_seconds,global_charged_seconds=total,global_cap_seconds=72*3600,tracked_bytes_plus_archive=disk_bytes,disk_cap_bytes=15*1024**3,new_generation_calls=0,new_simulator_batches_observed=1,remaining_documented_batches_not_authorized_after_FAIL=2)
lengths=sa['valid'].sum(axis=0)
r['denominators']=dict(source_cases=len(sr),source_clusters=len(set(x['source_id'] for x in sr)),offsets_per_source=3,source_complete_inputs=int((lengths==120).sum()),source_incomplete_inputs=int((lengths<120).sum()),source_completed=sum(x['completed'] for x in sr),baseline_cases=len(br),baseline_sources=len(set(x['source_id'] for x in br)),baseline_completed=sum(x['completed'] for x in br),baseline_terminated=int(ba['terminated'].sum()),baseline_joint_pass=sum(x['joint_pass'] for x in br),all_cases_retained=True)
r['runtime_integrity']=dict(initial_reset_gate_pass=bm['initial_reset_gate_pass'],initial_anchor_error_m=bm['initial_actual_reference_xy_max_m'],canonical_reference_error_m=float(np.abs(delta).max()),reference_velocity_error_m_s=vel_error,policy_observation_shape=list(trace['task_observation'].shape),reference_frames=list(map(int,ba['reference_frame'])),clock_contract='At action k reference k and observation k are recorded before action; actual k is recorded after that action completes. Next observation requests k+1. Same transition index is verified, not simultaneous pre-action reference/actual states.',original_endpoint_frame=119,original_endpoint_time_contract=bm['goal_provenance'],reference_velocity_rule=bm['reference_velocity_rule'],completion_fence_protocol=bm['completion_fence_protocol'],reset_scope='Seven exported kinematic tensors and four authoritative rigid-body caches; not hidden PhysX state.')
request=read(OUT/'requests/val_baseline_s11.json')
r['bindings']=dict(source=guard_batch(source,read(BASE/'requests/val_shared_s11_r0.json')),baseline=guard_batch(batch,request),receipt=receipt,observed_launcher_runtime_witness_sha256=sha(OUT/'runtime_witness.json'),versions_sha256=sha(OUT/'versions.json'),preparation_sha256=sha(OUT/'preparation.json'),source_original_goal_sha256=read(OUT/'preparation.json')['source_original_goal_sha256'],reference_hashes=read(OUT/'preparation.json')['reference_hashes'],source_case_order=read(OUT/'preparation.json')['source_case_order'])
r['all_failed_checks']=[k for k,v in r['checks'].items() if not v];r['status']='FAIL';r.pop('runtime_witness_sha256',None)
(OUT/'DOCUMENT_WITNESS.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
md=(OUT/'DOCUMENT_WITNESS.md').read_text(encoding='utf8')+'\nRetained failure detail: '+json.dumps(r['failure_detail'])+'\n\nDenominators: '+json.dumps(r['denominators'])+'\n\nBudget: '+json.dumps(r['budget'])+'\n\nRemaining failed checks: '+json.dumps(r['all_failed_checks'])+'\n\nActual launcher runtime witness hash (does not certify PASS): '+r['bindings']['observed_launcher_runtime_witness_sha256']+'\n\n'+r['runtime_integrity']['clock_contract']+'\n'
(OUT/'DOCUMENT_WITNESS.md').write_text(md,encoding='utf8')
print(json.dumps({'status':r['status'],'failed_checks':r['all_failed_checks'],'failure_detail':r['failure_detail'],'denominators':r['denominators'],'global_seconds':total,'reference_velocity_error':vel_error,'observed_launcher_runtime_witness_sha256':r['bindings']['observed_launcher_runtime_witness_sha256'],'json_sha256':sha(OUT/'DOCUMENT_WITNESS.json'),'md_sha256':sha(OUT/'DOCUMENT_WITNESS.md')},indent=2))
