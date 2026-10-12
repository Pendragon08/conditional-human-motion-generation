"""Independent read-only CPU audit after one observed documented invocation.
Writes only the witness report in this new result directory.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/support_reference_probe_v63_20261012'
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
sys.path.insert(0,str(ROOT/'scripts'))
from verify_execution_lineage_v39_20261012 import sha,read,guard_batch
from physics_guided_com_lineage_v24_20261012 import request
from physics_guided_metric_v23_20261012 import heading,wrap_angle

report=dict(status='FAIL',independent_witness=True,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    witness_agent='/root/support_reference_v63_document_witness',document='docs/support_reference_probe_v63_20261012.md',
    invocation='ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_support_reference_probe_v63_20261012.py --check',
    documented_invocation_count=1,prepare_invocations=0,run_invocations=0,retries=0,source_fixes=0,
    actual_invocation=dict(exit_code=0,sentinel='SUPPORT_REFERENCE_V63_DOCUMENT_CHECK_PASS',local_exec_session=77898),
    observed_processes=dict(launcher_pid=2881291,collector_pid=2881314,collector_child_pid=2881344,
        launcher_parent_pid=2881290,launcher_started_utc='2026-10-12 02:43:39',
        collector_started_utc='2026-10-12 02:43:39',collector_child_started_utc='2026-10-12 02:43:53',
        state_time_unix=1791773019.7298582,state_status='RUNNING',arm='baseline',
        resource_admission=dict(memory_used_mib=699,memory_total_mib=49140,utilization_percent=0),
        evidence='Live ps PID/PPID/lstart/args and RUNNING state observed during invocation; complete receipt and execution-time binding checked after exit.'),
    scope='Document execution and reference/data/runtime integrity only; synthetic simulation proxy, no efficacy, novelty, downstream or G1 conclusion.',
    training_calls=0,api_calls=0,overleaf_results_sync=False,checks={})
checks=report['checks']
def check(name,value):
    checks[name]=bool(value)
    assert value,name
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
try:
    v=read(OUT/'versions.json');p=read(OUT/'preparation.json')
    check('frozen_source_hashes',all(sha(ROOT/n)==h for n,h in v['files'].items()))
    check('source_manifest_hash',sha(OUT/'source_manifest.json')==v['source_manifest_sha256']==sha(BASE/'source_manifest.json'))
    check('thresholds_hash',sha(OUT/'frozen_thresholds.json')==v['thresholds_sha256']==sha(BASE/'frozen_thresholds.json'))
    check('parent_versions_hash',sha(BASE/'versions.json')==v['parent_versions_sha256'])
    check('all_reference_hashes',all(sha(OUT/'references'/(arm+'.npz'))==h for arm,h in p['reference_hashes'].items()))
    spec=read(ROOT/'configs/env-closd-baseline-20261010.json')
    spec_sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check('warm_runtime_spec_hash',spec_sha=='81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780')
    report['frozen_runtime']=dict(spec_sha256=spec_sha,python=spec['runtime'],overlay=spec['overlay'],base=spec['base'],packages_installed=0)
    weights={'controller_weights_sha256':ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth',
             'generator_weights_sha256':ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt'}
    check('frozen_model_weights',all(sha(path)==v[key] for key,path in weights.items()))
    source=BASE/'raw/val_shared_s11_r0'
    source_binding=guard_batch(source,read(BASE/'requests/val_shared_s11_r0.json'))
    check('source_execution_binding',source_binding==p['source_binding'])
    check('source_metrics_hash',sha(source/'metrics.json')==p['source_metrics_sha256'])
    sm=read(source/'metadata.json');sr=read(source/'metrics.json');sa=load_npz(source/'trajectories.npz')
    cases=sm['cases'];order=[r['case_id'] for r in sr]
    check('source_denominator',len(sr)==72 and len(cases)==72 and len(set(r['source_id'] for r in sr))==24)
    check('source_order_frozen',order==p['source_case_order']==[r['case_id'] for r in cases])
    source_counts=collections.Counter(r['source_id'] for r in cases)
    check('three_offsets_per_source',all(n==3 for n in source_counts.values()) and all(sorted(r['offset_m'] for r in cases if r['source_id']==sid)==[-.15,0.,.15] for sid in source_counts))
    manifest=read(OUT/'source_manifest.json');cfg=read(ROOT/'configs/physics_guided_20261011.json')
    wanted=[]
    for row in manifest['rows']:
        if row['split']=='val':
            for offset in cfg['offsets_m']:
                wanted.append(dict(row,offset_m=offset,goal_joint_body='Pelvis' if row['goal_joint']=='pelvis' else 'R_Wrist',case_id=row['clip_id']+':'+str(offset)))
    check('source_text_case_target_joint_order',wanted==cases)
    lengths=sa['valid'].sum(axis=0).astype(int)
    expected_baseline=np.empty((120,72,24,3),np.float32)
    for n,length in enumerate(lengths):
        check('source_valid_prefix_'+str(n),length>0 and np.array_equal(sa['valid'][:,n],np.arange(120)<length))
        expected_baseline[:length,n]=sa['actual'][:length,n]
        expected_baseline[length:,n]=sa['actual'][length-1,n]
    refs={arm:load_npz(OUT/'references'/(arm+'.npz')) for arm in ('baseline','unconstrained','endpoint_preserved')}
    ref=refs['baseline']['reference'];goal=np.asarray(sm['original_goal'],np.float32)
    check('baseline_is_first_simulated_actual_with_padding',np.array_equal(ref,expected_baseline))
    check('original_goal_hash',hashlib.sha256(goal.tobytes()).hexdigest()==p['source_original_goal_sha256']==sm['original_goal_sha256'])
    report['reference_invariants']={}
    for arm,x in refs.items():
        r=x['reference'];d=r.astype(np.float64)-ref.astype(np.float64)
        root_delta=d[:,:,0,:2]
        relative=(r-r[:,:,0:1])-(ref-ref[:,:,0:1])
        max_shift=float(np.linalg.norm(root_delta,axis=-1).max())
        roundoff=float(np.abs(relative).max())
        heading_delta=float(np.abs(wrap_angle(heading(r,sm['names'])-heading(ref,sm['names']))).max())
        check(arm+'_shape_finite',r.shape==(120,72,24,3) and np.isfinite(r).all())
        check(arm+'_target_exact',np.array_equal(x['original_goal'],goal))
        check(arm+'_first_61_frames_exact',np.array_equal(r[:61],ref[:61]))
        check(arm+'_height_exact',np.array_equal(r[:,:,:,2],ref[:,:,:,2]))
        check(arm+'_uniform_horizontal_translation',float(np.abs(d[:,:,:,:2]-root_delta[:,:,None,:]).max())<2e-6)
        check(arm+'_bounded_10cm',max_shift<=.100002)
        check(arm+'_relative_pose',roundoff<2e-6)
        check(arm+'_heading_roundoff',heading_delta<2e-6)
        check(arm+'_incomplete_inputs_unchanged',np.array_equal(r[:,lengths<120],ref[:,lengths<120]))
        if arm in ('baseline','endpoint_preserved'):check(arm+'_endpoint_exact',np.array_equal(r[-1],ref[-1]))
        report['reference_invariants'][arm]=dict(max_horizontal_translation_m=max_shift,endpoint_translation_max_m=float(np.linalg.norm(root_delta[-1],axis=-1).max()),relative_pose_roundoff_max_m=roundoff,heading_roundoff_max_rad=heading_delta,first_61_frames_exact=True,height_exact=True,all_72_targets_exact=True,incomplete_inputs_unchanged=True)
    batch=OUT/'raw/val_baseline_s11';bm=read(batch/'metadata.json');br=read(batch/'metrics.json');ba=load_npz(batch/'trajectories.npz');trace=load_npz(batch/'observation_trace.npz');receipt=read(batch/'receipt.json')
    expected=request(ROOT,OUT,'val','replay',11,0,dict(noise_round=0,replay_file=OUT/'references/baseline.npz'))
    check('actual_request',read(OUT/'requests/val_baseline_s11.json')==expected)
    binding=guard_batch(batch,expected)
    check('runtime_baseline_binding',binding==read(OUT/'runtime_witness.json')['result']['binding'])
    check('baseline_complete_denominator',len(br)==len(bm['cases'])==receipt['cases']==72)
    check('baseline_case_text_source_order',bm['cases']==wanted and [r['case_id'] for r in br]==order)
    check('baseline_original_goal_exact',np.array_equal(np.asarray(bm['original_goal'],np.float32),goal) and bm['original_goal_sha256']==p['source_original_goal_sha256'])
    check('baseline_original_heading_exact',bm['original_heading']==sm['original_heading'])
    check('baseline_target_and_heading_not_redefined',bm['proposed_goal']==bm['original_goal'] and np.array_equal(np.asarray(bm['proposed_heading'],np.float32),np.asarray(bm['original_heading'],np.float32)))
    check('baseline_no_generation',bm['generation_batch_calls']==bm['upstream_warmup_diffusion_calls']==bm['total_diffusion_calls_including_warmup']==0)
    check('reset_archive_hash',sha(batch/'initial_states.npz')==bm['initial_state_archive_sha256'])
    initial=load_npz(batch/'initial_states.npz')
    fields=tuple(bm['manual_reset_state_sha256'])
    check('seven_exported_reset_states_exact',len(fields)==7 and all(np.array_equal(initial['bootstrap_'+k],initial['first_action_'+k]) and hashlib.sha256(initial['first_action_'+k].tobytes()).hexdigest()==bm['first_action_state_sha256'][k]==bm['manual_reset_state_sha256'][k] for k in fields))
    check('native_reset_cache_gate',bm['initial_reset_gate_pass'] and all(x==0 for x in bm['initial_rigid_cache_consistency'].values()) and all(x==0 for x in bm['manual_reset_vs_first_action_state_max_difference'].values()))
    check('initial_anchor',bm['initial_actual_reference_xy_max_m']<.001)
    check('120_frame_clock',np.array_equal(ba['reference_frame'],np.arange(120)) and bm['episode_steps']==120 and abs(bm['dt']-1/30)<1e-6)
    canonical=(ref-ref[:,:,0:1])+ref[:,:,0:1]
    canonical_error=float(np.abs(ba['reference'][ba['valid']]-canonical[ba['valid']]).max())
    check('applied_reference_canonical_exact',canonical_error==0)
    check('observation_trace_hash',sha(batch/'observation_trace.npz')==bm['observation_trace_sha256'])
    check('reference_trace_same_applied_clock',np.array_equal(trace['ref_position'],ba['reference']))
    previous=np.concatenate([trace['ref_position'][:1],trace['ref_position'][:-1]],axis=0)
    check('reference_previous_successive_exact',np.array_equal(trace['ref_previous'],previous))
    vel=(trace['ref_position']-previous)/np.float32(bm['dt'])
    vel_error=float(np.abs(trace['ref_velocity']-vel).max())
    check('reference_velocity_successive_exact',vel_error==0 and not trace['ref_velocity'][0].any())
    check('raw_task_interface',trace['task_observation'].shape==(120,72,574) and bm['policy_observation_mode']==1 and bm['policy_observation_size']==574 and np.isfinite(trace['task_observation']).all())
    check('physx_cuda_cpu_export_completion_fence',not bm['sim_params']['use_gpu_pipeline'] and bm['sim_params']['physx_use_gpu'] and bm['completion_fence_calls']==242 and bm['contact_mutation_during_reward_max_N']==0)
    check('same_time_actual_state_shape',ba['actual'].shape==ba['reference'].shape==(120,72,24,3) and np.isfinite(ba['actual'][ba['valid']]).all())
    report['runtime_integrity']=dict(initial_reset_gate_pass=True,initial_anchor_error_m=bm['initial_actual_reference_xy_max_m'],canonical_reference_error_m=canonical_error,reference_velocity_error_m_s=vel_error,policy_observation_shape=list(trace['task_observation'].shape),reference_frames=list(range(120)),
        clock_contract='At action k, reference k and observation k are recorded before action; actual k is after that action completed, linked to reference k. Next observation requests k+1. Reference and actual do not claim to be simultaneous pre-action states.',
        original_endpoint_frame=119,original_endpoint_time_contract=bm['goal_provenance'],reference_velocity_rule=bm['reference_velocity_rule'],completion_fence_protocol=bm['completion_fence_protocol'],reset_scope='Seven exported kinematic tensors and four authoritative rigid-body caches; does not certify hidden PhysX state.')
    own=read(OUT/'budget_ledger.json');source_ledger=read(BASE/'budget_ledger.json');charged_source=[x for x in source_ledger if x['label']=='val_shared_s11_r0']
    check('single_charged_attempt',len(own)==1 and own[0]['label']=='val_baseline_s11' and own[0]['exit_code']==0 and own[0]['request_sha256']==sha(OUT/'requests/val_baseline_s11.json'))
    check('source_build_charged_not_zero',len(charged_source)==1 and charged_source[0]['seconds']>0 and charged_source[0]['exit_code']==0)
    prior=read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1]);dirs=[ROOT/n for n in prior['directories']]
    if OUT not in dirs:dirs.append(OUT)
    all_rows=[r for d in dirs if (d/'budget_ledger.json').exists() for r in read(d/'budget_ledger.json')]
    total=sum(r['seconds'] for r in all_rows);own_seconds=sum(r['seconds'] for r in own)
    disk_bytes=sum(f.stat().st_size for d in dirs for f in d.rglob('*') if f.is_file())+prior['archive_bytes']
    check('own_600_second_cap',own_seconds<600)
    check('global_72_gpu_hour_cap',total<72*3600)
    check('global_15gib_disk_cap',disk_bytes<15*1024**3)
    report['budget']=dict(charged_baseline_ledger=own,shared_source_native_generation_and_first_build_ledger=charged_source,shared_source_generation_batch_calls=sm['generation_batch_calls'],shared_source_cost_reused_not_zero=True,own_charged_seconds=own_seconds,own_remaining_seconds=600-own_seconds,global_charged_seconds=total,global_cap_seconds=72*3600,tracked_bytes_plus_archive=disk_bytes,disk_cap_bytes=15*1024**3,cpu_audit_does_not_generate_or_simulate=True,new_generation_calls=0,new_simulator_batches_observed=1,remaining_documented_batches=2)
    report['denominators']=dict(source_cases=72,source_clusters=24,offsets_per_source=3,source_complete_inputs=int((lengths==120).sum()),source_incomplete_inputs=int((lengths<120).sum()),source_completed=int(sum(r['completed'] for r in sr)),baseline_cases=72,baseline_sources=24,baseline_completed=sum(r['completed'] for r in br),baseline_terminated=int(ba['terminated'].sum()),baseline_joint_pass=sum(r['joint_pass'] for r in br),all_cases_retained=True)
    report['bindings']=dict(source=source_binding,baseline=binding,receipt=receipt,versions_sha256=sha(OUT/'versions.json'),preparation_sha256=sha(OUT/'preparation.json'),source_original_goal_sha256=p['source_original_goal_sha256'],reference_hashes=p['reference_hashes'],source_case_order=order)
    report['status']='PASS'
    report['runtime_witness_sha256']=sha(OUT/'runtime_witness.json')
except Exception as e:
    report['failure']=repr(e)
    report.pop('runtime_witness_sha256',None)

target=OUT/'DOCUMENT_WITNESS.json'
with target.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
lines=['# Independent V63 document witness','', 'Status: '+report['status'], '',
       'Executed the documented --check once. Exit 0; SUPPORT_REFERENCE_V63_DOCUMENT_CHECK_PASS observed. No prepare, --run, retry, source repair, API, training or Overleaf write.', '',
       'Observed launcher PID 2881291, collector PID 2881314, child PID 2881344; launcher/collector began 2026-10-12 02:43:39 UTC. Live process observation precedes complete receipt and execution-time hash binding.', '',
       'The JSON contains every actual check, source/reference/output binding, denominator, reset/interface/clock rule, and charged budget. Source construction cost is explicitly retained as nonzero.', '',
       'This witnesses execution/data integrity only. Tracking and surface metrics concern simulated synthetic proxies, not MoCap truth; no efficacy, novelty, downstream benefit or G1 conclusion.']
if report['status']=='PASS':
    lines+=['', 'Runtime witness SHA256: '+report['runtime_witness_sha256'],'',
        'Denominators: '+json.dumps(report['denominators'],ensure_ascii=False),'',
        'Reference invariants: '+json.dumps(report['reference_invariants'],ensure_ascii=False),'',
        'Charged baseline: %.9f seconds; own remaining %.9f / 600 seconds; global charged %.9f / 259200 seconds.'%(report['budget']['own_charged_seconds'],report['budget']['own_remaining_seconds'],report['budget']['global_charged_seconds']), '',
        report['runtime_integrity']['clock_contract']]
else:lines+=['','Failure: '+report['failure']]
with (OUT/'DOCUMENT_WITNESS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'runtime_witness_sha256':report.get('runtime_witness_sha256'),'failure':report.get('failure'),'checks':len(checks),'json_sha256':sha(target),'md_sha256':sha(OUT/'DOCUMENT_WITNESS.md')},indent=2))
