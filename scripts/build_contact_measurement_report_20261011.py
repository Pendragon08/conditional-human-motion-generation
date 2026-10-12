"""Render source-grounded diagnostic status; no paper or Overleaf writes."""
from pathlib import Path
import hashlib
import html
import json
from replay_report_certificate_v30_20261012 import certify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_measurement_review_20261011'
OUT.mkdir(exist_ok=True)
def read(path):
    file=ROOT/path
    return json.loads(file.read_text(encoding='utf8')) if file.exists() else None

v8=read('results/contact_sampling_v8_20261011/sampling_summary.json')
box=read('results/contact_force_fixture_20261011/repeat_comparison.json')
v10=read('results/contact_sensor_v10_20261011/sampling_summary.json')
box_v2=read('results/contact_force_fixture_v2_20261011/repeat_comparison.json')
box_v3=read('results/contact_force_fixture_v3_20261012/repeat_comparison.json')
v11=read('results/contact_cpu_export_v11_20261012/sampling_summary.json')
quality11=read('results/contact_cpu_export_v11_20261012/quality_diagnostic.json')
validation12=read('results/contact_cpu_validation_v12_20261012/validation_summary.json')
slip13=read('results/contact_slip_diagnostic_v13_20261012/diagnostic.json')
point16=read('results/contact_point_fixture_v16_20261012/first_attempt/summary.json')
point17=read('results/contact_point_human_v17_20261012/state.json')
velocity19=read('results/velocity_reference_fixture_v19_20261012/first_attempt/summary.json')
point20=read('results/contact_point_human_v20_20261012/state.json')
com21=read('results/contact_com_metric_v21_20261012/diagnostic.json')
articulated22=read('results/articulated_contact_fixture_v22_20261012/first_attempt/summary.json')
validation24=read('results/contact_com_validation_v24_20261012/validation_summary.json')
numerics25=read('results/metric_runtime_diagnostic_v25_20261012/diagnostic.json')
runtime26=read('results/contact_com_runtime_v26_20261012/runtime_check.json')
replay27=read('results/fixed_reference_com_v27_20261012/replay_summary.json')
formal28=read('results/condition_feedback_formal_v28_20261012/queue_state.json')
progress_paths=sorted((ROOT/'results/condition_feedback_formal_v28_20261012').glob('progress_snapshot_*.json'))
formal_progress=read(str(progress_paths[-1].relative_to(ROOT))) if progress_paths else None
if formal_progress:
    assert formal_progress['source_sha256']==hashlib.sha256((ROOT/'scripts/snapshot_condition_feedback_v28_20261012.py').read_bytes()).hexdigest()
    assert formal_progress['final_gate'] is False and formal_progress['innovation_confirmed'] is False
component_paths=sorted((ROOT/'results/condition_feedback_components_v32_20261012').glob('diagnostic_*.json'))
components=read(str(component_paths[-1].relative_to(ROOT))) if component_paths else None
if components:
    assert components['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_condition_feedback_components_v32_20261012.py').read_bytes()).hexdigest()
    assert components['live_pipeline_mutated'] is False and components['thresholds_changed'] is False
    assert components['simulation_calls']==0 and components['generation_calls']==0
    assert components['innovation_confirmed'] is False
repeat_paths=sorted((ROOT/'results/feedback_repeatability_diagnostic_20261012').glob('diagnostic_*.json'))
repeat_diagnostic=read(str(repeat_paths[-1].relative_to(ROOT))) if repeat_paths else None
if repeat_diagnostic:
    assert repeat_diagnostic['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_feedback_repeatability_20261012.py').read_bytes()).hexdigest()
    assert repeat_diagnostic['simulation_calls']==0 and repeat_diagnostic['c2_simulation_efficacy_tested'] is False
formal_summary=read('results/condition_feedback_formal_v28_20261012/summary.json')
formal_gate=read('results/condition_feedback_formal_v28_20261012/progression_gate.json')
formal_audit=read('results/condition_feedback_formal_v28_20261012/EXPERIMENT_AUDIT.json')
mocap_control=read('results/mocap_reference_control_v33_20261012/diagnostic_summary.json')
mocap_witness=read('results/mocap_reference_control_v33_20261012/DOCUMENT_WITNESS_AUDIT.json')
attribution_paths=sorted((ROOT/'results/original_reference_attribution_v35_20261012').glob('diagnostic_*.json'))
original_attribution=read(str(attribution_paths[-1].relative_to(ROOT))) if attribution_paths else None
binding_paths=sorted((ROOT/'results/mocap_cache_binding_v36_20261012').glob('certificate_*.json'))
strict_binding=read(str(binding_paths[-1].relative_to(ROOT))) if binding_paths else None
event_paths=sorted((ROOT/'results/endpoint_event_diagnostic_v37_20261012').glob('diagnostic_*.json'))
endpoint_event=read(str(event_paths[-1].relative_to(ROOT))) if event_paths else None
lineage_paths=sorted((ROOT/'results/execution_lineage_guard_v39_20261012').glob('certificate_*.json'))
strict_lineage=read(str(lineage_paths[-1].relative_to(ROOT))) if lineage_paths else None
transfer_paths=sorted((ROOT/'results/construction_replay_transfer_v40_20261012').glob('diagnostic_*.json'))
transfer=read(str(transfer_paths[-1].relative_to(ROOT))) if transfer_paths else None
state_paths=sorted((ROOT/'results/raw_replay_state_diagnostic_v41_20261012').glob('diagnostic_*.json'))
replay_state=read(str(state_paths[-1].relative_to(ROOT))) if state_paths else None
observation_trace=read('results/reference_observation_v42_20261012/summary.json')
observation_witness=read('results/reference_observation_v42_20261012/DOCUMENT_WITNESS.json')
velocity_trace=read('results/reference_velocity_v43_20261012/summary.json')
velocity_witness=read('results/reference_velocity_v43_20261012/DOCUMENT_WITNESS.json')
original_velocity=read('results/original_reference_velocity_v45_20261012/summary.json')
original_velocity_witness=read('results/original_reference_velocity_v45_20261012/DOCUMENT_WITNESS.json')
native_paths=sorted((ROOT/'results/native_velocity_summary_v47_20261012').glob('summary_*.json'))
native_control=read(str(native_paths[-1].relative_to(ROOT))) if native_paths else None
native_witness=read('results/native_reference_velocity_v46_20261012/DOCUMENT_WITNESS.json')
support_paths=sorted((ROOT/'results/support_velocity_diagnostic_v49_20261012').glob('diagnostic_*.json'))
support_diagnostic=read(str(support_paths[-1].relative_to(ROOT))) if support_paths else None
analytical52=read('results/surface_proxy_calibration_v52_20261012/first_attempt/summary.json')
analytical52_witness=read('results/surface_proxy_calibration_v52_20261012/DOCUMENT_WITNESS.json')
paired51_summary=read('results/paired_feedback_validation_v51_20261012/summary.json')
paired51_state=read('results/paired_feedback_validation_v51_20261012/state.json')
paired51_progress=read('results/paired_feedback_validation_v51_20261012/progress.json')
paired51_preview=read('results/paired_feedback_validation_v51_20261012/previews/manifest.json')
paired53_audit=read('results/paired_feedback_analysis_v53_20261012/EXPERIMENT_AUDIT.json')
paired53_paths=sorted((ROOT/'results/paired_feedback_analysis_v53_20261012').glob('analysis_*.json'))
paired53=read(str(paired53_paths[-1].relative_to(ROOT))) if paired53_paths else None
if paired53:
    assert paired53['source_sha256']==hashlib.sha256((ROOT/'scripts/analyze_paired_feedback_v51_20261012.py').read_bytes()).hexdigest()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in paired53['input_hashes'].items())
paired54_paths=sorted((ROOT/'results/paired_feedback_analysis_v54_20261012').glob('analysis_*.json'))
paired54=read(str(paired54_paths[-1].relative_to(ROOT))) if paired54_paths else None
paired54_binding=read('results/paired_feedback_analysis_v54_20261012/execution_binding_certificate.json')
paired54_audit=read('results/paired_feedback_analysis_v54_20261012/EXPERIMENT_AUDIT.json')
selected55_paths=sorted((ROOT/'results/selected_construction_binding_v55_20261012').glob('certificate_*.json'))
selected55=read(str(selected55_paths[-1].relative_to(ROOT))) if selected55_paths else None
native56=read('results/native_manifold_v56_20261012/summary.json')
native56_witness=read('results/native_manifold_v56_20261012/DOCUMENT_WITNESS.json')
native61_receipt=read('results/native_manifold_v56_20261012/continuation_receipt_v61.json')
native57_audit=read('results/native_manifold_analysis_v57_20261012/EXPERIMENT_AUDIT.json')
support63=read('results/support_reference_probe_v63_20261012/summary.json')
support63_witness=read('results/support_reference_probe_v63_20261012/DOCUMENT_WITNESS.json')
support63_audit=read('results/support_reference_probe_v63_20261012/EXPERIMENT_AUDIT.json')
support63_addendum=read('results/support_reference_probe_v63_20261012/NUMERICAL_CONTRACT_ADDENDUM.json')
support63_post_audit=read('results/support_reference_probe_v63_20261012/POST_RUN_AUDIT.json')
support63_scope=read('results/support_reference_probe_v63_20261012/SCOPE_CLARIFICATION.json')
remaining_screen=read('results/remaining_candidate_screen_20261012/component_screen.json')
remaining_screen_audit=read('results/remaining_candidate_screen_20261012/EXPERIMENT_AUDIT.json')
leg_ik_preflight=read('results/support_leg_ik_20261012/preflight.json')
leg_ik_witness=read('results/support_leg_ik_20261012/DOCUMENT_WITNESS.json')
leg_ik_continuation=read('results/support_leg_ik_20261012/CONTINUATION_DOCUMENT_WITNESS.json')
leg_ik_adjudicated_witness=read('results/support_leg_ik_20261012/ADJUDICATED_DOCUMENT_WITNESS.json')
leg_ik_summary=read('results/support_leg_ik_20261012/summary.json')
leg_ik_post_audit=read('results/support_leg_ik_20261012/POST_RUN_AUDIT.json')
child_binding_check=read('results/child_input_binding_20261012/cpu_checks.json')
controller_state_analysis=read('results/controller_state_diagnosis_v2_20261012/analysis.json')
controller_state_audit=read('results/controller_state_diagnosis_v2_20261012/EXPERIMENT_AUDIT.json')
controller_state_witness=read('results/controller_state_diagnosis_v2_20261012/DOCUMENT_WITNESS.json')
controller_state_original_failure=read('results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json')
if controller_state_analysis:
    assert controller_state_analysis['same_algorithm_different_training_states']
    assert not controller_state_analysis['candidate_efficacy_test'] and controller_state_analysis['confirmed_innovations']==0
    controller_versions=read('results/controller_state_diagnosis_v2_20261012/versions.json')
    for name,digest in controller_versions['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
if child_binding_check:
    assert child_binding_check['candidate_efficacy_test'] is False and child_binding_check['gpu_runtime_ready'] is False
    for name,digest in child_binding_check['scripts'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
leg_ik_terminal=read('results/support_leg_ik_20261012/EXECUTOR_TERMINAL_OBSERVATION.json')
leg_ik_velocity_paths=sorted((ROOT/'results/support_leg_ik_20261012').glob('velocity_components_[0-9]*.json'))
leg_ik_analysis_paths=sorted((ROOT/'results/support_leg_ik_20261012').glob('analysis_[0-9]*.json'))
leg_ik_analysis=read(str(leg_ik_analysis_paths[-1].relative_to(ROOT))) if leg_ik_analysis_paths else None
if leg_ik_analysis:
    if leg_ik_analysis['source_sha256']!=hashlib.sha256((ROOT/'scripts/analyze_support_leg_ik_20261012.py').read_bytes()).hexdigest():
        raise RuntimeError('Frozen leg IK analyzer changed')
    if leg_ik_analysis['candidate_efficacy_test'] or leg_ik_analysis['confirmed_innovation']:
        raise RuntimeError('Engineering diagnosis cannot count as an innovation')
if remaining_screen:
    if remaining_screen['candidate_efficacy_test'] or remaining_screen['new_simulator_calls'] or remaining_screen['new_generation_calls']:
        raise RuntimeError('Retrospective component screen cannot be promoted to efficacy')
    for name,digest in remaining_screen['input_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Component-screen input binding changed: '+name)
support63_paths=sorted((ROOT/'results/support_reference_probe_v63_20261012').glob('analysis_[0-9]*.json'))
support63_analysis=read(str(support63_paths[-1].relative_to(ROOT))) if support63_paths else None
native57_paths=sorted((ROOT/'results/native_manifold_analysis_v57_20261012').glob('diagnostic_*.json'))
native57=read(str(native57_paths[-1].relative_to(ROOT))) if native57_paths else None
if paired54:
    assert paired54['source_sha256']==hashlib.sha256((ROOT/'scripts/analyze_paired_feedback_v54_20261012.py').read_bytes()).hexdigest()
    assert paired54['bootstrap_resamples']==5000
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in paired54['input_hashes'].items())
if selected55:
    assert selected55['source_sha256']==hashlib.sha256((ROOT/'scripts/verify_selected_construction_v55_20261012.py').read_bytes()).hexdigest()
    assert selected55['simulation_calls']==0 and selected55['original_files_changed'] is False
    assert all(e['exact_reference_reconstruction'] and e['selection_labels_reconstructed_from_bound_raw'] for e in selected55['entries'])
if native57:
    assert native57['source_sha256']==hashlib.sha256((ROOT/'scripts/analyze_native_manifold_v57_20261012.py').read_bytes()).hexdigest()
    assert native57['metrics_replaced'] is False and native57['simulation_calls']==0
if support63_analysis:
    assert support63_analysis['source_sha256']==hashlib.sha256((ROOT/'scripts/analyze_support_reference_probe_v63_20261012.py').read_bytes()).hexdigest()
    assert support63_analysis['confirmed_innovation'] is False and support63_analysis['thresholds_changed'] is False
if analytical52:
    assert analytical52['status']=='PASS_ANALYTICAL_CONTROLS' and analytical52['checks']==24
    assert all(x['check_pass'] for x in analytical52['cases'])
    assert analytical52['source_sha256']==hashlib.sha256((ROOT/'scripts/calibrate_surface_proxy_v52_20261012.py').read_bytes()).hexdigest()
    assert analytical52['parent_source_sha256']==hashlib.sha256((ROOT/'scripts/calibrate_surface_proxy_v50_20261012.py').read_bytes()).hexdigest()
    assert analytical52_witness and analytical52_witness['status']=='PASS'
    assert analytical52_witness['summary_sha256']==hashlib.sha256((ROOT/'results/surface_proxy_calibration_v52_20261012/first_attempt/summary.json').read_bytes()).hexdigest()
    assert analytical52['simulation_calls']==0 and analytical52['gpu_calls']==0
if support_diagnostic:
    assert support_diagnostic['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_support_velocity_v49_20261012.py').read_bytes()).hexdigest()
    assert support_diagnostic['parent_source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_support_velocity_v48_20261012.py').read_bytes()).hexdigest()
    assert support_diagnostic['simulation_calls']==0 and support_diagnostic['thresholds_changed'] is False
    assert all(arm['frozen_slip_fractions_exactly_reproduced'] for arm in support_diagnostic['arms'].values())
if native_control:
    assert native_control['source_sha256']==hashlib.sha256((ROOT/'scripts/summarize_native_velocity_v47_20261012.py').read_bytes()).hexdigest()
    assert native_control['simulation_calls']==0 and native_control['original_failure_preserved'] is True
    assert native_witness and native_witness['status']=='FAIL' and native_witness['documented_launcher_exit_code']==1
    assert native_witness['checks']['two_successful_charged_measurement_batches'] is True
    assert native_witness['checks']['all72metrics_exact_under_bound_runtime'] is True
    for label,batch in (('val_native_s11','native_construction'),('val_native_replay_s11','native_fixed_reference')):
        for name,digest in native_control['batches'][batch]['output_hashes'].items():
            if name.endswith('.npz'): continue
            path=ROOT/'results/native_reference_velocity_v46_20261012/public/raw'/label/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
if original_velocity:
    assert original_velocity['generation_calls']==0 and original_velocity['thresholds_changed'] is False
    assert original_velocity_witness and original_velocity_witness['status']=='PASS'
    for name,digest in original_velocity['output_hashes'].items():
        if name.endswith('.npz'): continue
        path=ROOT/'results/original_reference_velocity_v45_20261012/public/raw/val_original_replay_s11'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
quality_paths=sorted((ROOT/'results/reference_quality_v44_20261012').glob('diagnostic_*.json'))
velocity_quality=read(str(quality_paths[-1].relative_to(ROOT))) if quality_paths else None
if velocity_trace:
    assert velocity_trace['attempts']==72 and velocity_trace['candidate_efficacy'] is False
    assert velocity_witness and velocity_witness['status']=='PASS' and velocity_witness['efficacy_inferred'] is False
    version=read('results/reference_velocity_v43_20261012/versions.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in version['files'].items())
if velocity_quality:
    assert velocity_quality['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_reference_quality_v44_20261012.py').read_bytes()).hexdigest()
    assert velocity_quality['simulation_calls']==0 and velocity_quality['thresholds_changed'] is False
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in velocity_quality['input_hashes'].items())
if observation_trace:
    assert observation_trace['status']=='COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC' and observation_trace['attempts']==72
    assert observation_trace['source_sha256']==hashlib.sha256((ROOT/'scripts/compare_reference_observation_v42_20261012.py').read_bytes()).hexdigest()
    assert observation_trace['candidate_efficacy'] is False and observation_trace['historical_scores_changed'] is False
for diagnostic,source in ((transfer,'diagnose_construction_replay_transfer_v40_20261012.py'),
                          (replay_state,'diagnose_raw_replay_state_v41_20261012.py')):
    if diagnostic:
        assert diagnostic['source_sha256']==hashlib.sha256((ROOT/'scripts'/source).read_bytes()).hexdigest()
        assert diagnostic['simulation_calls']==0 and diagnostic['generation_calls']==0
        assert diagnostic['historical_scores_changed'] is False and diagnostic['innovation_confirmed'] is False
if strict_lineage:
    assert strict_lineage['status']=='PASS' and strict_lineage['checked_batches']==74
    assert strict_lineage['retroactive_lineages_accepted']==0 and strict_lineage['historical_results_changed'] is False
    assert strict_lineage['source_sha256']==hashlib.sha256((ROOT/'scripts/verify_execution_lineage_v39_20261012.py').read_bytes()).hexdigest()
if strict_binding:
    assert strict_binding['status']=='PASS' and strict_binding['old_scores_changed'] is False
    assert strict_binding['source_sha256']==hashlib.sha256((ROOT/'scripts/verify_mocap_cache_binding_v36_20261012.py').read_bytes()).hexdigest()
    assert sum(row['attempts'] for row in strict_binding['batches'].values())==216
    assert all(row['all_metric_rows_exactly_recomputed'] for row in strict_binding['batches'].values())
if endpoint_event:
    assert endpoint_event['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_endpoint_event_v37_20261012.py').read_bytes()).hexdigest()
    assert endpoint_event['historical_scores_changed'] is False and endpoint_event['acceptance_rule_selected'] is False
    assert endpoint_event['binding_certificate_sha256']==hashlib.sha256(binding_paths[-1].read_bytes()).hexdigest()
resource_paths=sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))
resource_status=read(str(resource_paths[-1].relative_to(ROOT))) if resource_paths else None
if resource_status:
    assert resource_status['source_sha256']==hashlib.sha256((ROOT/'scripts/summarize_authorized_round_20261012.py').read_bytes()).hexdigest()
if mocap_control:
    assert mocap_control['status']=='COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC'
    assert mocap_control['generator_sample_calls']==0 and mocap_control['thresholds_changed'] is False
    assert mocap_witness and mocap_witness['verdict']=='PASS'
    for label,split in (('val_mocap_replay_s11','validation'),('test_mocap_replay_s11','test')):
        for name,digest in mocap_control[split]['output_hashes'].items():
            if name.endswith('.npz'): continue  # Licensed arrays remain on the server.
            file=ROOT/'results/mocap_reference_control_v33_20261012/public/raw'/label/name
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest
if original_attribution:
    assert original_attribution['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_original_reference_quality_v35_20261012.py').read_bytes()).hexdigest()
    assert original_attribution['simulation_calls']==0 and original_attribution['generation_calls']==0
    assert original_attribution['summary_sha256']==hashlib.sha256((ROOT/'results/mocap_reference_control_v33_20261012/diagnostic_summary.json').read_bytes()).hexdigest()
if formal_summary:
    assert formal_gate==formal_summary['gate']
    assert len(formal_summary['summary'])==10
    assert all(row['attempts']==432 and row['sources']==48 for row in formal_summary['summary'].values())
    version=read('results/condition_feedback_formal_v28_20261012/versions.json')
    assert version['files']['scripts/physics_guided_analyze.py']==hashlib.sha256((ROOT/'scripts/physics_guided_analyze.py').read_bytes()).hexdigest()
replay_certificate=certify(ROOT,replay27)
cached_guard=read('results/fixed_reference_com_v27_20261012/cached_guard_recheck_v30_20261012.json')
cached_guard_pass=(cached_guard is not None and cached_guard.get('status')=='PASS' and len(cached_guard.get('entries',[]))==8 and all(row.get('pass') is True for row in cached_guard['entries']) and cached_guard.get('source_sha256')==hashlib.sha256((ROOT/'scripts/verify_cached_measurement_guards_v30_20261012.py').read_bytes()).hexdigest())
replay_interface_pass=(replay_certificate['status']=='PASS' and cached_guard_pass and runtime26 is not None and runtime26.get('status')=='PASS' and velocity19 is not None and velocity19.get('status')=='PASS' and articulated22 is not None and articulated22.get('status')=='PASS')
probe=read('results/condition_feedback_probe_v2_20261011/probe_summary.json')
inputs=[ROOT/name for name in ('results/contact_sampling_v8_20261011/sampling_summary.json',
 'results/contact_force_fixture_20261011/repeat_comparison.json',
 'results/contact_sensor_v10_20261011/sampling_summary.json',
 'results/contact_force_fixture_v2_20261011/repeat_comparison.json',
 'results/contact_force_fixture_v3_20261012/repeat_comparison.json',
 'results/contact_cpu_export_v11_20261012/sampling_summary.json',
 'results/contact_cpu_export_v11_20261012/quality_diagnostic.json',
 'results/contact_cpu_validation_v12_20261012/validation_summary.json',
 'results/contact_slip_diagnostic_v13_20261012/diagnostic.json',
 'results/contact_point_fixture_v16_20261012/first_attempt/summary.json',
 'results/contact_point_human_v17_20261012/state.json',
 'results/velocity_reference_fixture_v19_20261012/first_attempt/summary.json',
 'results/contact_point_human_v20_20261012/state.json',
 'results/contact_com_metric_v21_20261012/diagnostic.json',
 'results/articulated_contact_fixture_v22_20261012/first_attempt/summary.json',
 'results/contact_com_validation_v24_20261012/validation_summary.json',
 'results/metric_runtime_diagnostic_v25_20261012/diagnostic.json',
 'results/contact_com_runtime_v26_20261012/runtime_check.json',
 'results/fixed_reference_com_v27_20261012/replay_summary.json',
 'results/fixed_reference_com_v27_20261012/cached_guard_recheck_v30_20261012.json',
 'results/condition_feedback_formal_v28_20261012/queue_state.json',
 'results/condition_feedback_probe_v2_20261011/probe_summary.json') if (ROOT/name).exists()]
if component_paths: inputs.append(component_paths[-1])
if repeat_paths: inputs.append(repeat_paths[-1])
if attribution_paths: inputs.append(attribution_paths[-1])
if binding_paths: inputs.append(binding_paths[-1])
if event_paths: inputs.append(event_paths[-1])
if lineage_paths: inputs.append(lineage_paths[-1])
if transfer_paths: inputs.append(transfer_paths[-1])
if state_paths: inputs.append(state_paths[-1])
for name in ('summary.json','DOCUMENT_WITNESS.json','versions.json','budget_ledger.json'):
    path=ROOT/'results/reference_observation_v42_20261012'/name
    if path.exists(): inputs.append(path)
if resource_paths: inputs.append(resource_paths[-1])
for name in ('summary.json','DOCUMENT_WITNESS.json','versions.json','budget_ledger.json'):
    path=ROOT/'results/reference_velocity_v43_20261012'/name
    if path.exists(): inputs.append(path)
if quality_paths: inputs.append(quality_paths[-1])
if native_paths: inputs.append(native_paths[-1])
if support_paths: inputs.append(support_paths[-1])
for directory,names in (
    ('surface_proxy_calibration_v50_20261012',('DOCUMENT_WITNESS.json',)),
    ('surface_proxy_calibration_v52_20261012',('first_attempt/summary.json','DOCUMENT_WITNESS.json')),
    ('paired_feedback_validation_v51_20261012',('summary.json','state.json','progress.json','versions.json','runtime_witness.json','DOCUMENT_WITNESS.json','budget_ledger.json','launcher_receipt.json','previews/manifest.json')),
    ('paired_feedback_analysis_v53_20261012',('EXPERIMENT_AUDIT.json',)),
    ('paired_feedback_analysis_v54_20261012',('execution_binding_certificate.json','EXPERIMENT_AUDIT.json')),
    ('native_manifold_v56_20261012',('summary.json','DOCUMENT_WITNESS.json','runtime_witness.json','budget_ledger.json')),
    ('native_manifold_analysis_v57_20261012',('EXPERIMENT_AUDIT.json',)),
    ('remaining_candidate_screen_20261012',('component_screen.json','EXPERIMENT_AUDIT.json')),
    ('support_leg_ik_20261012',('preflight.json','DOCUMENT_WITNESS.json','DOCUMENT_WITNESS_FROZEN_ANALYSIS_BINDING.json','preparation.json','versions.json','analysis_plan_lock.json','budget_ledger.json','SCALAR_CONTRACT_DIAGNOSIS.json','continuation_preflight.json','CONTINUATION_DOCUMENT_WITNESS.json','CONTINUATION_CONTRACT_ADDENDUM.json','adjudicated_preflight.json','ADJUDICATED_DOCUMENT_WITNESS.json','summary.json','launch_receipt.json','pipeline_finally_receipt.json','POST_RUN_AUDIT.json')),
    ('support_reference_probe_v63_20261012',('summary.json','DOCUMENT_WITNESS.json','runtime_witness.json','budget_ledger.json','EXPERIMENT_AUDIT.json','NUMERICAL_CONTRACT_ADDENDUM.json','analysis_plan_lock.json','CONTINUATION_DOCUMENT_WITNESS_V65.json','continuation_v65_receipt.json','POST_RUN_AUDIT.json','SCOPE_CLARIFICATION.json','PARENT_EXIT_OBSERVATION.json'))):
    for name in names:
        path=ROOT/'results'/directory/name
        if path.exists(): inputs.append(path)
if paired53_paths: inputs.append(paired53_paths[-1])
if paired54_paths: inputs.append(paired54_paths[-1])
if selected55_paths: inputs.append(selected55_paths[-1])
if native57_paths: inputs.append(native57_paths[-1])
if support63_paths: inputs.append(support63_paths[-1])
if leg_ik_analysis_paths: inputs.append(leg_ik_analysis_paths[-1])
if leg_ik_velocity_paths: inputs.append(leg_ik_velocity_paths[-1])
if leg_ik_terminal: inputs.append(ROOT/'results/support_leg_ik_20261012/EXECUTOR_TERMINAL_OBSERVATION.json')
if child_binding_check: inputs.append(ROOT/'results/child_input_binding_20261012/cpu_checks.json')
for dirname in ('controller_state_diagnosis_20261012','controller_state_diagnosis_v2_20261012'):
    for name in ('DOCUMENT_WITNESS.json','versions.json','preparation.json','budget_ledger.json','runtime_witness.json',
                 'analysis.json','EXPERIMENT_AUDIT.json','EXECUTOR_TERMINAL_OBSERVATION.json'):
        path=ROOT/'results'/dirname/name
        if path.exists():inputs.append(path)
for state in ('multitask','t2m','unfinetuned'):
    for name in ('loaded_input_binding.json','loaded_checkpoint_binding.json','metrics.json','metadata.json','lineage.json','receipt.json'):
        path=ROOT/'results/controller_state_diagnosis_v2_20261012/raw'/state/name
        if path.exists():inputs.append(path)
for name in ('DOCUMENT_WITNESS.json','versions.json','budget_ledger.json','interface_summary.json'):
    path=ROOT/'results/native_reference_velocity_v46_20261012'/name
    if path.exists(): inputs.append(path)
for name in ('summary.json','DOCUMENT_WITNESS.json','versions.json','budget_ledger.json'):
    path=ROOT/'results/original_reference_velocity_v45_20261012'/name
    if path.exists(): inputs.append(path)
for name in ('diagnostic_summary.json','DOCUMENT_WITNESS_AUDIT.json'):
    path=ROOT/'results/mocap_reference_control_v33_20261012'/name
    if path.exists(): inputs.append(path)
for name in ('summary.json','progression_gate.json','versions.json','EXPERIMENT_AUDIT.json'):
    path=ROOT/'results/condition_feedback_formal_v28_20261012'/name
    if path.exists(): inputs.append(path)
status=dict(innovation_candidates=3,preliminarily_tested=1,not_yet_experimentally_tested=2,
 confirmed_innovations=0,measurement_gate='CALIBRATED_PROXY_AND_FIXED_REPLAY_PASS' if replay_interface_pass else 'FIXED_REPLAY_NOT_CERTIFIED',g1='NOT_STARTED_BY_GATE',
 replay_export_certificate=replay_certificate,
 cached_guard_recheck=cached_guard,
 formal_comparison_snapshot=formal28,
 formal_completed_arm_snapshot=formal_progress,
 failure_component_diagnostic=components,
 formal_analysis=formal_summary,formal_integrity_audit=formal_audit,
 repeat_feedback_diagnostic=repeat_diagnostic,
 original_reference_control=mocap_control,original_reference_attribution=original_attribution,
 strict_original_reference_cache_binding=strict_binding,endpoint_event_diagnostic=endpoint_event,
 strict_execution_lineage_guard=strict_lineage,
 construction_replay_transfer=transfer,raw_replay_state_diagnostic=replay_state,
 reference_observation_diagnostic=observation_trace,reference_observation_witness=observation_witness,
 reference_velocity_repair=velocity_trace,reference_velocity_witness=velocity_witness,
 repaired_interface_failure_breakdown=velocity_quality,
 original_reference_repaired_interface=original_velocity,
 original_reference_repaired_interface_witness=original_velocity_witness,
 native_corrected_interface_control=native_control,native_corrected_interface_witness=native_witness,
 support_velocity_diagnostic=support_diagnostic,
 analytical_contact_controls=analytical52,analytical_contact_witness=analytical52_witness,
 corrected_feedback_validation_summary=paired51_summary,corrected_feedback_validation_state=paired51_state,
 corrected_feedback_validation_progress=paired51_progress,
corrected_feedback_readonly_analysis=paired53,corrected_feedback_integrity_audit=paired53_audit,
 corrected_feedback_analysis_amendment=paired54,current_execution_binding=paired54_binding,
 current_construction_selection_reconstruction=selected55,analysis_amendment_audit=paired54_audit,
 passive_native_manifold_summary=native56,passive_native_manifold_witness=native56_witness,
 native_material_point_diagnostic=native57,
 native_contact_continuation_receipt=native61_receipt,native_material_point_audit=native57_audit,
 reference_side_support_probe=support63,reference_side_support_probe_witness=support63_witness,
 reference_side_support_probe_analysis=support63_analysis,reference_side_support_probe_audit=support63_audit,
 reference_side_support_probe_numerical_addendum=support63_addendum,
 reference_side_support_probe_post_run_audit=support63_post_audit,reference_side_support_probe_scope=support63_scope,
 remaining_candidate_component_screen=remaining_screen,remaining_candidate_component_screen_audit=remaining_screen_audit,
 support_leg_ik_preflight=leg_ik_preflight,support_leg_ik_independent_witness=leg_ik_witness,
 support_leg_ik_continuation_witness=leg_ik_continuation,support_leg_ik_summary=leg_ik_summary,
 support_leg_ik_adjudicated_witness=leg_ik_adjudicated_witness,
 support_leg_ik_raw_analysis=leg_ik_analysis,
 support_leg_ik_post_run_audit=leg_ik_post_audit,
 prospective_child_input_binding_check=child_binding_check,
 controller_training_state_diagnosis=controller_state_analysis,
 controller_training_state_audit=controller_state_audit,
 controller_training_state_document_witness=controller_state_witness,
 controller_training_state_original_failure=controller_state_original_failure,
 support_leg_ik_external_terminal_observation=leg_ik_terminal,
 resource_status=resource_status,
 full_efficacy_comparisons_completed=1 if formal_summary else 0,
 downstream='NOT_STARTED_BY_GATE',overleaf_results_sync=False,
 slip13=slip13,point16=point16,point17=point17,velocity19=velocity19,point20=point20,com21=com21,articulated22=articulated22,validation24=validation24,numerics25=numerics25,runtime26=runtime26,replay27=replay27,
 v8=v8,box=box,v10=v10,box_v2=box_v2,box_v3=box_v3,v11=v11,quality11=quality11,validation12=validation12,
 input_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')

rows=[('V11 CPU导出人体重复','新设置重复性通过，质量尚不通过',f"重复运动、接触力和动作均一致；与旧设置实际位置最大差{v11['cpu_export_vs_v10_state_max']['actual']:.3f}m，不混合旧结果。12输入脚滑代理通过0。" if v11 else '未同步'),
 ('V6','保留错误调用','队列实际调用了V5；不能作为完成等待机制的验证'),
 ('V7','重置一致性失败','等待放在重置末尾时，导出刚体状态与官方重置缓存不一致；未启动轨迹采集'),
 ('V8','记录完整，接触测量不通过',f"两次各12输入×120帧，运动学完全一致；净接触力最大差{v8['contact_force']:.2f}N" if v8 else '结果未同步'),
 ('解析几何测试','仅几何计算通过','5个已知几何测试，不是人体物理执行成功证明'),
 ('物理箱体测试','两次校准通过，接口重复性有差异',f"状态与约束传感器完全一致；净接触力最大差{box['repeat_maximum_differences']['net_contact_force']:.2f}N" if box else '结果未同步'),
 ('箱体V2缓冲区清空','未解决重复性',f"状态、传感器不变，净接触力仍差{box_v2['repeat_maximum_differences']['net_contact_force']:.2f}N；修复假设不获支持" if box_v2 else '结果未同步'),
 ('箱体V3 CPU导出','箱体校验和重复性通过',f"状态和传感器相对原版本不变，重复净接触力差{box_v3['repeat_maximum_differences']['net_contact_force']:.6f}N；尚未验证人体环境" if box_v3 else '结果未同步'),
 ('V9','保留首次失败','适配器字符串断言失败，仿真和传感器校验没有执行'),
 ('V10','人体被动传感器记录复核',('传感器重复性：'+str(v10.get('sensor_repeatable_at_1mN'))+'；不等于真实接触力') if v10 else '尚未得到完整两次复核结果')]
rows.extend([
 ('V13 432次离线几何诊断','不支持最低角点是主要原因','原指标精确重算；诊断性选择附近最慢角点也仅1/432过线，不替代评分。'),
 ('V14/V15 接触接口适配','保留两次失败','V14变量覆盖；V15字段命名不匹配。失败并不证明API不可用。'),
 ('V16 接触接口物体校准','已通过' if point16 else '未同步','3312条原生接触、0错误；已知箱体状态/力/传感器与V3完全一致。'),
 ('V17 人体接触记录','采集完成，校验程序失败' if point17 else '未同步','新增绑定键与旧校验器不兼容；独立核查53764条接触0错误、所有状态/动作与V11一致。'),
 ('V19 速度参考位置','独立校准通过' if velocity19 else '未同步','线速度对应质心；6cm偏置、3rad/s旋转时，原点解释误差0.18006m/s，质心解释误差0.0000703m/s。'),
 ('V20 人体记录接口修复',point20['status'] if point20 else '未同步','保留三键绑定；密集接触文件另由元信息哈希绑定；原阈值与物理不变。'),
 ('V21 质心速度修正','离线计算修正完成' if com21 else '未同步','修正旋转杠杆臂；12输入仍脚滑代理0/12、联合0/12。旧评分保留，不算创新。'),
 ('V22 已知关节体支撑/滑动/腾空',articulated22['status'] if articulated22 else '待复核','使用原判据验证正确静止、明确滑动和腾空，结果不能推广为所有滚动接触或真实力。'),
 ('V25 数值环境差异','原因确认' if numerics25 else '待同步','保存时使用NumPy 1.23.5，旧复算使用1.24.4；统一到采集环境后，72条指标差异为0，分类不变。'),
 ('V26 复算环境修复',runtime26['status'] if runtime26 else '待同步','独立见证保留旧失败及11个输出绑定，复用已完成72条，无新仿真、阈值或容差修改。'),
 ('V27 固定参考回放',replay27['status'] if replay27 else '待完成','回放相同输入的原生成参考及物理轨迹参考，所有来源与构建失败保留；非候选效力实验。')])
table=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
p=probe['summary'] if probe else {}
probe_table=''.join(f'<tr><td>{html.escape(name)}</td><td>{row["attempts"]}</td><td>{row["endpoint_error_m"]*100:.2f}</td><td>{row["condition_pass"]}</td><td>{row["physical_pass"]}</td><td>{row["joint_pass"]}</td></tr>' for name,row in p.items())
validation_text='质心速度修正后的全24验证来源、3生成种子、两种前缀验证尚未完成。'
if validation24:
    validation_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key in ('mode','seed','inputs','completed','condition_pass','physical_pass','joint_pass'))+'</tr>' for row in validation24['entries'])
    validation_text='<p>质心速度修正后已完成432次验证尝试：24来源×3偏移×3种子×2前缀。它是验证集诊断，不能称为每方法432测试输入的正式实验。原目标、阈值不变，失败保留。</p><table><tr><th>前缀</th><th>种子</th><th>尝试</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th></tr>'+validation_rows+'</table><p>阶段决定：'+html.escape(validation24['decision'])+'。物理质量未通过与测量程序未通过是不同问题；此表不证明候选机制无效，更不证明创新有效。</p>'
replay_text='<p>固定参考回放尚未完成。</p>'
if replay27:
    replay_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key in ('reference','attempted','completed','condition_pass','physical_pass','tracking_pass','combined_construction_replay_joint_pass','construction_terminated'))+'</tr>' for row in replay27['entries'])
    replay_text='<table><tr><th>固定参考</th><th>尝试</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>跟踪合格</th><th>构建＋回放联合合格</th><th>构建终止</th></tr>'+replay_rows+'</table><p>所有72输入都保留。构建终止后补齐的序列不能因回放成功而恢复联合成功资格。这是单种子接口与质量诊断，不是新算法对比。</p>'
measurement_text='已知正负对照通过，固定参考回放尚待完成。'
formal_text='<p>C1完整对比尚未启动。</p>'
if formal28:
    formal_text='<p>服务器采集状态快照：'+html.escape(formal28['status'])+'；批次：'+html.escape(formal28.get('label','未提供'))+'。完整协议包含10个对比与消融配置，每配置432测试输入。状态快照不代替实时进程检查，也不是完成结果；尚无新的效力结论。</p>'
if formal_progress:
    snapshot_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in
        ('arm','seed','attempts','condition_pass','physical_pass','joint_pass','early_terminated'))+'</tr>'
        for row in formal_progress['entries'])
    formal_text='<p>已完成回放批次：'+str(formal_progress['complete_replay_arm_seed_batches'])+'/'+str(formal_progress['expected_replay_arm_seed_batches'])+'。快照采集时已核实原进程存在，正在运行 '+html.escape(formal_progress['queue_state'].get('label','未提供'))+'。这是带时间戳的部分结果，不表示打开报告时仍在该批次，也不是正式结论。</p><table><tr><th>方法</th><th>种子</th><th>输入</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th><th>提前终止</th></tr>'+snapshot_rows+'</table><p>当前已完成的第一个种子中，误差补偿的条件合格为34/144，native_best为24/144；但二者联合合格均为0/144。因此条件改善不能当作物理质量或训练价值改善。其余种子尚待完成；所有判据保持不变。</p>'
component_text='<p>完整来源失败分量尚未同步。</p>'
if components:
    selected_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (row['arm'],row['seed'],row['attempts'],row['component_pass_counts']['condition'],
         row['component_pass_counts']['physical'],row['complete_construction_and_replay_physical_pass'],
         row['physical_pass_but_construction_failed'],row['component_pass_counts']['joint']))+'</tr>'
        for row in components['entries'] if row['arm'] in ('native_best','proposed','planned_prefix'))
    component_text='<p>只读核对了'+str(components['completed_arm_seed_batches'])+'/30个完成回放批次和'+str(len(components['construction_batches']))+'个候选构建批次；各批次均绑定输入、选择记录及轨迹哈希，不改阈值或队列。</p><table><tr><th>方法</th><th>种子</th><th>输入</th><th>条件合格</th><th>回放物理代理合格</th><th>构建与回放均完成且物理合格</th><th>物理合格但构建失败</th><th>联合合格</th></tr>'+selected_rows+'</table><p>物理代理合格数可能包含构建失败后补齐的参考，不能把这类样本计为有效生成。首个种子的三个物理代理合格案例均属于构建失败；第二个种子的计划前缀消融出现一个构建与回放均完成的物理合格案例，但未同时满足全部条件与跟踪判据。</p><p>三个种子中，误差补偿的条件合格数为34、33、31，native_best为24、23、25，计划前缀消融为33、33、38，各分母144。计划前缀消融并未始终逊于实际前缀；所有方法联合合格为零，因此不能将部分条件改善解释为物理质量或训练价值改善。</p>'
repeat_text='<p>重复执行前提尚未核对。</p>'
if repeat_diagnostic:
    rows_repeat=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (row['seed'],row['repeat'],row['all_arrays_equal'],row['repeated_complete'],row['position_agreement'],row['heading_agreement'],row['completed_cases_goal_changes'],row['completed_cases_heading_changes']))+'</tr>'
        for row in repeat_diagnostic['entries'])
    repeat_text='<p>以下核对使用已完成C1零更新对照，不新增仿真。原始目标和噪声相同，提议朝向存在约5.92×10⁻⁸ rad记录差异，不称为新的严格同输入C2重复实验。</p><table><tr><th>种子</th><th>对照</th><th>保存轨迹完全一致</th><th>重复均完成</th><th>位置一致性通过</th><th>朝向一致性通过</th><th>完整案例门控改变位置</th><th>完整案例门控改变朝向</th></tr>'+rows_repeat+'</table><p>目前完整案例中的一致性门控全部接受，有无门控产生的更新相同；差异仅在未完成案例。重复一致不能证明系统误差正确。C2质量守护及新增候选效力尚未测试，不增加已验证创新数量。</p>'
if formal_summary:
    complete_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (arm,row['attempts'],row['joint_pass'],row['completed'],round(row['endpoint_error_m'],4),row['early_terminated']))+'</tr>'
        for arm,row in formal_summary['summary'].items())
    formal_text='<p>完整分析已生成：10个方法配置，每配置432输入、48原始来源、3生成种子。所有构建失败和回放失败均保留。独立完整性审计状态：'+html.escape(str(formal_audit.get('verdict','未提供') if formal_audit else '尚未完成'))+'。分析文件的产生不等于创新验收通过。</p><table><tr><th>方法</th><th>输入</th><th>联合合格</th><th>回放完成</th><th>平均终点误差(m)</th><th>提前终止</th></tr>'+complete_rows+'</table><p>验收：'+html.escape(formal_gate['reason'])+'。G1和下游训练仍须遵守原有门槛；不得用部分条件改善代替联合质量收益。</p><p><a href="../condition_feedback_formal_v28_20261012/experiment_report.html">完整对比与消融报告</a> · <a href="../condition_feedback_formal_v28_20261012/summary.json">配对来源统计</a></p>'
mocap_text='<p>原始MoCap参考回放尚未完成。</p>'
if mocap_control:
    mocap_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (split,offset,row['attempts'],row['completed'],row['condition_pass'],row['physical_pass'],row['tracking_pass'],row['joint_pass']))+'</tr>'
        for split in ('validation','test') for offset,row in mocap_control[split]['by_offset'].items())
    mocap_text='<p>原始HumanML3D参考经过冻结表示转换，直接交给同一跟踪器；不调用动作生成，不修改原目标或阈值。全部72验证和144测试输入保留；±15cm目标并非原参考的任务，须分别看零偏移结果。独立接口见证PASS仅认证记录，不证明动作物理可行。</p><table><tr><th>集合</th><th>偏移(m)</th><th>输入</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>跟踪合格</th><th>联合合格</th></tr>'+mocap_rows+'</table><p>零偏移测试48来源中，物理代理合格2、联合合格0。当前固定跟踪器／表示转换／碰撞表面代理的组合，也会对原始参考产生低合格率。因此C1的零收益不是生成器独立失效的证明；原始参考也不是机器人动力学真值。该诊断不作为第四个创新、不改变C1否定结果、不降低后续验收门槛。</p>'
if original_attribution:
    attribute_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (row['split'],row['offset'],row['attempts'],row['reference_continuity_pass'],row['component_pass_counts']['continuity'],row['complete_reference_continuity_pass_but_executed_fail']))+'</tr>'
        for row in original_attribution['groups'])
    mocap_text+='<table><tr><th>集合</th><th>偏移范围</th><th>输入</th><th>参考连续性符合旧阈值</th><th>执行连续性合格</th><th>参考合格但完整执行后不合格</th></tr>'+attribute_rows+'</table><p>这只是同一时间窗口的运动学连续性诊断。未把人体参考当作接触力真值，也未新增物理成功标准；足部滑移仍使用原表面代理。应先定位跟踪／表示／代理的影响，再安排C2效力实验及G1、下游训练。</p>'
if strict_binding:
    mocap_text+='<p>V36只读严格缓存校验通过：固定独立见证与最终摘要的校验和，核对来源、阈值、模型和数据，并从72验证与144测试轨迹精确重算全部216行指标。重置七项状态与参考时钟一致。此证书补齐当前缓存绑定检查，不修改原V33源文件或历史评分，也不是候选机制的有效性证据。</p><p><a href="../mocap_cache_binding_v36_20261012/'+binding_paths[-1].name+'">缓存绑定与精确复算证书</a></p>'
if endpoint_event:
    event_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key in ('split','offset','attempts','deadline_position_satisfied_but_closest_timing_fails','physically_and_tracking_qualified_subset'))+'</tr>' for row in endpoint_event['groups'])
    mocap_text+='<h3>到达时刻的事件含义</h3><p>原指标计算生成段内最接近目标的时刻距最终帧的差，并不等同于第一次进入10厘米范围，也不等同于最终时刻是否在目标范围内。三个事件应分别命名，不能把最接近时刻的偏移直接解释为迟到。</p><table><tr><th>集合</th><th>偏移</th><th>输入</th><th>最终位置达标但原时刻判据失败</th><th>其中完成且物理／跟踪合格</th></tr>'+event_rows+'</table><p>零偏移测试的17/48条存在上述差异，其中2条满足原物理与跟踪判据。这是事件定义的诊断，不是新的成功率；没有重算验收、改阈值或选择更有利的事件。历史C1结论保持未获支持。第一次诊断对无生成段的提前终止案例报错，失败记录保留，修复后的诊断包含全部样本。</p><p><a href="../endpoint_event_diagnostic_v37_20261012/'+event_paths[-1].name+'">逐样本事件定义诊断</a></p>'
if formal_audit:
    formal_text+='<h3>独立完整性审计</h3><p>原控制实现审计为FAIL：旧启动器允许在缺少执行时来源记录时，对已有完成文件事后补写记录。当前74批数据未触发该缺陷，全部具有执行时绑定，数据与否定结论相符。到达时刻定义另有WARN。审计是同模型家族的独立会话复核，语义结论属于暂定；不能将它写成原协议全部通过。</p><p><a href="../condition_feedback_formal_v28_20261012/EXPERIMENT_AUDIT.md">完整审计与原始证据</a></p>'
if strict_lineage:
    formal_text+='<p>V39新增严格只读保护，拒绝缺失、事后生成、请求变更或输出变化的来源记录，锚定两类见证与原版本。隔离正负样例通过；当前74批全部重新核对通过。此保护不修改旧启动器、不改变原审计FAIL或历史成绩；后续新队列必须集成该保护并重新见证后方可运行。</p><p><a href="../execution_lineage_guard_v39_20261012/'+lineage_paths[-1].name+'">74批严格来源校验证书</a></p>'
if transfer:
    component_text+='<h3>构建合格不等于参考重放合格</h3><p>测试构建的6048次尝试中有7次满足原联合判据。跨方法选择中有8个构建合格的实例（含同一来源重复选择，不能称为8条独立动作），重放后全部失去联合合格，均未通过物理质量判据。这说明C3的收益校准应采用固定参考重放结果，不能将构建时的合格标签直接当作可训练数据价值。验证构建144次尝试联合合格为0，不能使用测试标签替代缺失的验证校准。</p><p><a href="../construction_replay_transfer_v40_20261012/'+transfer_paths[-1].name+'">构建与重放逐例核对</a></p>'
if replay_state:
    component_text+='<h3>重置一致性与数值差异</h3><p>3种子、每种子144输入的原生成与原参考重放配对中，15项初始状态、刚体属性与足部几何均逐元素一致，实际应用的参考位置符合相同的原生坐标变换。首步动作存在最大约2.98×10⁻⁷的差异，首步实际位置最大差约0.267毫米；后续有效帧的差异增大。诊断分别报告共同有效输入数与包含失效槽位的差异，不能把提前终止后的数组用于有效执行比较。尚不能将差异归因于重置、速度接口或生成器；下一步需要记录跟踪器实际接收的参考速度及观测。</p><p><a href="../raw_replay_state_diagnostic_v41_20261012/'+state_paths[-1].name+'">初始状态与有效帧数值诊断</a></p>'
if observation_trace:
    component_text+='<h3>实际参考速度与原始观测记录</h3><p>V42新增两批验证集仿真，各保留24原始来源×3目标偏移，共72输入；记录参考前一帧、参考速度与动作前原始任务观测。生成衔接帧（第60帧）的应用位置完全一致，但前一帧最大差0.0741m、参考速度最大差2.2230m/s。第0帧已有约8.94×10⁻⁷m/s速度差，第1帧原始任务观测最大差0.04373；因此不能把后续失效仅归因于衔接帧。原始任务观测不是策略归一化后的输入。生成侧2次提前终止、重放侧1次，均保留在分母。</p><p>独立文档执行见证核对新记录与原V28验证构建的全部9个轨迹数组逐元素一致，说明本次被动记录未改变该批原执行结果；见证仅覆盖当前诊断，不认证C2/C3效力。后续需要单独修复参考速度的时间语义并校验，任何新结果与历史结果分开记录。</p><p><a href="../reference_observation_v42_20261012/summary.json">逐帧接口诊断</a> · <a href="../reference_observation_v42_20261012/DOCUMENT_WITNESS.md">独立执行见证</a></p>'
if replay_interface_pass:
    measurement_text='重置、坐标、时间、数值复算、已知正负对照及固定参考回放接口已通过。足部指标仍是碰撞表面滑移代理，不能称为真实接触力或所有滚动接触的真值。人体物理质量与候选效力另行判断。'
if velocity_trace:
    component_text+='<h3>参考速度一致性修复：V43</h3><p>单独使用连续两帧实际送入跟踪器的位置计算参考速度，首帧设为零，保留全部原目标与判据。两批72组验证输入在120帧、8582个共同有效帧—案例对上，参考位置、前一帧、参考速度、原始任务观测与实际轨迹全部逐元素一致；完整数组也一致。生成参考位置与V42完全相同。原生浮点倒数乘法能精确重建保存的速度，CPU直接除法最多约9.54×10⁻⁷的舍入差异已保留记录。本结果只确认当前单种子验证对的接口一致性，不能推广为三种子效力或机器人执行成功。</p><p><a href="../reference_velocity_v43_20261012/DOCUMENT_WITNESS.md">独立完整帧见证</a> · <a href="../reference_velocity_v43_20261012/summary.json">接口结果</a></p>'
if velocity_quality:
    q=velocity_quality['batches']['val_planned_s11']; counts=q['pass_counts']
    component_text+=f'<h3>修复后失败分量：V44</h3><p>72输入中完成{counts["completed"]}，位置判据通过{counts["position"]}，朝向{counts["heading"]}，时刻{counts["time"]}，穿透{counts["penetration"]}，连续性{counts["continuity"]}，跟踪{counts["tracking"]}；足部滑移代理通过{counts["slip_proxy"]}，联合合格{q["joint_pass"]}。27组仅在物理判据中的滑移代理不合格，45组同时不符合滑移代理与连续性。滑移是碰撞表面代理，不是真实接触点滑移；不能从该零分直接认定生成动作普遍不可行。下一项诊断使用原始MoCap派生参考和修复后的接口，不采样生成器，帮助检查质量基线。</p><p><a href="../reference_quality_v44_20261012/{quality_paths[-1].name}">固定判据逐例分解</a></p>'
if original_velocity:
    c=original_velocity['pass_counts']
    component_text+=f'<h3>原始参考在修复接口下的校准：V45</h3><p>72组原始MoCap派生验证参考，生成器采样0次；完成{c["completed"]}，条件合格{c["condition_pass"]}，物理代理合格{c["physical_pass"]}，跟踪合格{c["tracking_pass"]}，联合合格{c["joint_pass"]}。零偏移24组联合合格1组。原始目标和判据不变，失败全部保留。结果证明这组设置存在非零合格实例，但通过率偏低；原始参考也不是经认证的可执行真值，不能据此单独区分源动作缺陷、控制器限制或代理误判。该校准不计为创新有效性。</p><p><a href="../original_reference_velocity_v45_20261012/summary.json">校准结果</a> · <a href="../original_reference_velocity_v45_20261012/DOCUMENT_WITNESS.md">独立见证</a></p>'
if native_control:
    c=native_control['batches']['native_fixed_reference']
    component_text+=f'<h3>原生CLoSD对照与汇总失败：V46/V47</h3><p>两批仿真各72组均完成，但原V46启动器在汇总阶段错误读取回放元数据中不存在的生成噪声字段，退出1；原文档执行验收为FAIL，历史RUNNING文件为过期状态，不能据此声称进程仍运行。单独的V47只读CPU分析通过构建元数据核对配对噪声，不修改V46、不重复仿真。独立复核确认120帧的轨迹、原始观测与初始状态完全一致，原生构建与回放均条件合格{c["condition_pass"]}、物理合格{c["physical_pass"]}、跟踪合格{c["tracking_pass"]}、联合合格{c["joint_pass"]}。这仍是既有基线的单种子验证校准，不是C2/C3有效性结论。</p><p><a href="../native_reference_velocity_v46_20261012/DOCUMENT_WITNESS.md">原执行失败及批次复核</a> · <a href="../native_velocity_summary_v47_20261012/{native_paths[-1].name}">独立只读汇总</a></p>'
if support_diagnostic:
    a=support_diagnostic['arms']['planned'];b=a['angular_bins_rad_s']['0.0-0.25']
    component_text+=f'<h3>支撑速度代理的探索性分解：V49</h3><p>计划参考回放中共有{a["proxy_violations_total"]}个支撑帧—刚体观测超过冻结速度阈值；其中生成衔接帧60附近±2帧为{a["proxy_violations_near_frame60"]}个。低角速度（小于0.25rad/s）的{b["support_observations"]}个支撑观测中，仅{b["above_frozen_slip_speed"]}个超阈值；高角速度观测中超阈值明显更多。原始MoCap参考也出现相似关联。这是事后描述性分组，未改接受规则；接触点滚动、碰撞表面近似和真实滑移尚不能由该代理分开，不能宣称足部旋转造成滑移或已找到创新收益。V48首次分析遗漏原生质心参数注入而失败，原代码与失败记录保留，V49单独采用逐案例原生质心后精确重现全部原滑移代理分数。后续须在已知接触运动下检查代理的判别边界，再设计物理精化对比。</p><p><a href="../support_velocity_diagnostic_v49_20261012/{support_paths[-1].name}">支撑速度分解</a> · <a href="../support_velocity_diagnostic_v48_20261012/attempt1_failure.json">保留的首次分析失败</a></p>'
if analytical52:
    component_text+='<h3>解析接触控制：V52</h3><p>四个实际碰撞盒的24项解析控制通过：静止、正反向理想边缘滚动未被判为滑动，0.2m/s平移及滚动叠加平移被识别，腾空无支撑。独立核对使用显式旋转矩阵与所有并列最低顶点；但输入接触力是假定值，角度是独立参数样本。这是公式级合成控制，不是仿真、真实接触点或人体无滑移的证据。V50首个CPU命令因文件路径错误失败，0项控制执行，失败另行保留。高角速度关联本身不足以推翻原代理，历史分数与阈值不变。</p><p><a href="../surface_proxy_calibration_v52_20261012/DOCUMENT_WITNESS.md">独立解析见证</a> · <a href="../surface_proxy_calibration_v50_20261012/DOCUMENT_WITNESS.md">原路径失败</a></p>'
if paired51_state:
    component_text+='<h3>修复接口后的配对反馈验证：V51</h3><p>此轮为验证集阶段，每方法24来源×3偏移×3种子，合计216输入；两方法共用实际前缀与逐候选配对噪声，每输入均为3候选，并独立回放选择出的精化参考。与旧V28协议分开，不混合分数，也不计为第四项创新。最近同步的状态为 '+html.escape(paired51_state['status'])+'；该文件状态不是当前进程存活的证明。</p>'
    if paired51_summary:
        component_text+='<p>完整验证摘要：'+html.escape(json.dumps(paired51_summary['totals'],ensure_ascii=False))+'。只有验证集结果，仍需按独立测试和原验收门判断；不等于有效创新或完成G1／下游实验。</p>'
    reports=paired51_summary['reports'] if paired51_summary else (paired51_progress or {}).get('reports',{})
    if reports:
        paired_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (label,report['cases'],*[report['replay'][k] for k in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')]))+'</tr>'
            for label,report in sorted(reports.items()))
        component_text+='<table><tr><th>方法／种子</th><th>输入</th><th>回放完成</th><th>条件合格</th><th>物理代理合格</th><th>跟踪合格</th><th>回放联合合格</th></tr>'+paired_rows+'</table><p>表中是回放分量计数；构建阶段失败不能因补齐后的回放完成而恢复成功资格，完整端到端结果另作核对。</p>'
    if paired53:
        component_text+=f'<p>V53只读分析：端到端联合合格增益{paired53["gain_percentage_points"]:.2f}个百分点；按原始来源聚类的95%区间为{html.escape(str(paired53["source_cluster_95_ci_percentage_points"]))}个百分点。所有构建失败保留在分母。此区间仅适用于24个验证来源；若全部合格为零，退化的[0,0]区间不证明总体效应不存在。</p>'
        component_text+='<p>独立测试、语义与多样性护栏、G1和下游训练尚未完成，数值分析不认证创新。<a href="../paired_feedback_analysis_v53_20261012/'+paired53_paths[-1].name+'">分析与输入哈希</a></p>'
        component_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (arm,*[paired53['component_pass_counts'][arm][k] for k in ('position','heading','time','surface_slip_proxy','penetration','continuity','construction_complete')]))+'</tr>' for arm in ('feedback','random'))
        component_text+='<table><tr><th>方法（各216输入）</th><th>位置</th><th>朝向</th><th>时刻</th><th>滑移代理</th><th>穿透</th><th>连续性</th><th>构建完整</th></tr>'+component_rows+'</table>'
    component_text+='<p><strong>指标来源说明：</strong>跟踪误差相对于生成／精化的合成参考（synthetic proxy）计算，不是相对于真实MoCap或真实机器人真值。物理质量包含模拟接触代理，跟踪误差降低不单独证明物理可执行性或训练收益。独立审计指出原始跟踪字段的标注不足；原文件和分数保留，新分析明确区分评价来源。<a href="../../docs/paired_feedback_v51_evaluation_provenance_20261012.md">补充说明</a></p>'
    if paired53_audit:
        component_text+='<p>独立审计结论：'+html.escape(str(paired53_audit.get('overall_verdict',paired53_audit.get('verdict','Not verified'))))+'（同模型家族、暂定）。<a href="../paired_feedback_analysis_v53_20261012/EXPERIMENT_AUDIT.md">完整审计与原始标注问题</a></p>'
    if paired54:
        component_text+='<h4>追加复核与历史审计边界</h4><p>V54按冻结配置使用5000次来源聚类自助采样，核对21个批次的执行请求、输出与观测文件。V55另从执行时绑定的候选原始数据精确重建选择标签及参考，并与回放所用运动哈希核对；当前端到端联合合格仍为0/216和0/216。两份证据应共同引用。它们是追加式复核，不追溯修复原始合成参考标注不足、旧元数据矛盾或历史选择记录绑定缺口，不认证创新。<a href="../paired_feedback_analysis_v54_20261012/'+paired54_paths[-1].name+'">V54分析</a>'
        if selected55_paths:
            component_text+=' · <a href="../selected_construction_binding_v55_20261012/'+selected55_paths[-1].name+'">V55选择重建</a>'
        component_text+='</p>'
    if paired54_audit:
        component_text+='<p>独立审计整体结论：'+html.escape(str(paired54_audit.get('overall_verdict',paired54_audit.get('verdict','Not verified'))))+'。整体包含保留的历史问题，不表示当前零收益结果已通过创新验收。<a href="../paired_feedback_analysis_v54_20261012/EXPERIMENT_AUDIT.md">审计全文</a></p>'
    if native56:
        component_text+='<h4>原生接触材料点诊断</h4><p>V56首批及独立V61续跑共完成两批各72输入的被动接触导出，与V51对应的33项轨迹、观测、初始状态、传感器数组及原指标完全一致。原V56主入口、V58及V60续跑入口的三次仿真前失败均保留，不能将续跑成功称为原入口成功。此诊断只增加观测，生成器采样零次，不增加已验证创新数。<a href="../native_manifold_v56_20261012/summary.json">一致性证据</a> · <a href="../native_manifold_v56_20261012/continuation_receipt_v61.json">续跑收据</a></p>'
    if native57:
        native_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            arm,r['matched_support_bodyframes'],r['proxy_matched_speed']['above_frozen_speed'],
            r['native_matched_min_speed']['above_frozen_speed'],r['native_matched_max_speed']['above_frozen_speed'],
            r['proxy_above_but_all_native_points_at_or_below_speed']))+'</tr>' for arm,r in native57['arms'].items())
        component_text+='<p>V57比较原最低碰撞点代理与同物理时刻的模拟接触材料点速度。下表是相关的身体—时刻观测，不能视为独立试验；不替换原评分，不是真实接触真值。</p><table><tr><th>方法</th><th>配对支撑观测</th><th>原代理超速</th><th>全部接触点超速</th><th>至少一接触点超速</th><th>代理超速但全部接触点未超速</th></tr>'+native_rows+'</table><p><a href="../native_manifold_analysis_v57_20261012/'+native57_paths[-1].name+'">完整逐样本诊断</a>。只有与实际接触一致的观测才能用于定位剩余滑动问题；这些诊断不证明控制器独立不可行，也不能直接作为C2/C3有效性证据。</p>'
        component_text+='<p>两组约49%的配对支撑观测中，所有报告接触点的水平速度均超过冻结的5厘米/秒阈值。因此最低碰撞点选择差异不足以单独解释当前质量地板；这仍是单种子、模拟环境中的诊断，不将失效归因于某一个模块。</p>'
        if native57_audit:
            component_text+='<p>接触诊断独立审计：'+html.escape(str(native57_audit.get('overall_verdict',native57_audit.get('verdict','Not verified'))))+'（同模型家族，暂定）。<a href="../native_manifold_analysis_v57_20261012/EXPERIMENT_AUDIT.md">审计全文与边界</a></p>'
    if support63_witness:
        component_text+='<h4>参考修正实验的校验记录</h4><p>原参考回放已完成72输入。初次独立见证为'+html.escape(support63_witness['status'])+'，记录原样保留；后续数值复核状态为'+html.escape(str(support63_addendum.get('status','未完成') if support63_addendum else '未完成'))+'。复核只按此前冻结的参考变换容差和实际CUDA速度表达式核对保存数据，不重跑基线、不修改评分判据。<a href="../support_reference_probe_v63_20261012/DOCUMENT_WITNESS.md">原始见证</a> · <a href="../support_reference_probe_v63_20261012/NUMERICAL_CONTRACT_ADDENDUM.md">独立补充核验</a>。</p>'
    if support63_analysis:
        support_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (
            arm,r['inputs'],r['counts']['completed'],r['counts']['condition_pass'],r['counts']['physical_pass'],r['counts']['joint_pass'],
            r['available_metric_means'].get('surface_slip_proxy_fraction'),r['available_metric_means'].get('endpoint_error_m')))
            +'</tr>' for arm,r in support63_analysis['reports'].items())
        component_text+='<h4>参考侧支撑阶段修正对照</h4><p>固定一个共同构建来源，比较原参考、有限水平根平移修正及加入末帧位姿保持的修正。24验证来源、3原始偏移、单种子，每组72输入；所有失败保留，原始目标、时间、模型及评分不变。新生成零次，各方法各有一次回放；共同构建的实际成本另行保留，不能称为免费。该对照是有限参数化诊断，不增加已确认创新数。</p><table><tr><th>参考</th><th>输入</th><th>完成</th><th>条件合格</th><th>物理合格</th><th>联合合格</th><th>平均滑移代理比例</th><th>平均终点误差(m)</th></tr>'+support_rows+'</table><p>诊断决策：'+html.escape(support63_analysis['decision'])+'。末帧参考位姿保持不等于实际终点准确；根平移不能解决局部关节、足部旋转或完整动力学。<a href="../support_reference_probe_v63_20261012/'+support63_paths[-1].name+'">完整分析</a>。V62的CPU测例失败保留，V63不重写其记录。</p>'
        if support63_audit:
            component_text+='<p>运行前独立审计（保留的历史快照）：'+html.escape(str(support63_audit.get('overall_verdict',support63_audit.get('verdict','Not verified'))))+'。<a href="../support_reference_probe_v63_20261012/EXPERIMENT_AUDIT.md">审计记录</a></p>'
        if support63_scope:
            component_text+='<p>划分边界：本轮是内部验证子集，包含上游HumanML3D官方train的19来源/57输入及官方test的5来源/15输入，未评估本研究内部的独立测试集。不能报告为未触碰的官方test泛化，也未逐源确认预训练模型是否见过这些动作。<a href="../support_reference_probe_v63_20261012/SCOPE_CLARIFICATION.md">划分说明</a></p>'
        if support63_post_audit:
            component_text+='<p>事后独立审计：'+html.escape(str(support63_post_audit.get('overall_verdict',support63_post_audit.get('verdict','Not verified'))))+'（同模型家族，暂定）。原始失败、运行前审计与事后审计分别保留。<a href="../support_reference_probe_v63_20261012/POST_RUN_AUDIT.md">完整事后核查</a>。父进程退出记录来自工具观察，未伪装成服务器自动生成的退出收据。</p>'
    if remaining_screen:
        screen_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (r['seed'],r['inputs'],r['construction_candidates_per_input'],r['choices'].get('0',0),sum(v for k,v in r['choices'].items() if k!='0'),r['cases_with_observed_joint_qualified_candidate']))+'</tr>' for r in remaining_screen['reports'])
        component_text+='<h4>剩余候选机制的已有记录筛查</h4><p>仅在C1既有三候选记录上调用C2的逐分量质量守护，不是C2预注册的“重复执行＋一次修正”实验。未新增生成或仿真，216组输入全数保留。最终197组保留初次候选，19组改选；这说明守护并非完全不动作，但不能证明其提高执行质量。每组已有候选中均无物理或联合合格样本，因此从这个有限候选库重新选择，不能产生联合合格收益；不能据此推断未来新候选也一定无效。</p><table><tr><th>种子</th><th>输入</th><th>已有候选/输入</th><th>保留初次</th><th>改选</th><th>存在联合合格候选的输入</th></tr>'+screen_rows+'</table><p>C2非零联合合格前提仍未满足，C3也没有验证集联合修复收益用于校准；二者正式效力试验仍未运行。不能把本次回顾性筛查算为新仿真实验、预算节约或已验证创新。<a href="../remaining_candidate_screen_20261012/component_screen.json">全部选择和拒绝理由</a>。</p>'
        if remaining_screen_audit:
            component_text+='<p>筛查独立审计：'+html.escape(str(remaining_screen_audit.get('overall_verdict',remaining_screen_audit.get('verdict','Not verified'))))+'（同模型家族，暂定）。<a href="../remaining_candidate_screen_20261012/EXPERIMENT_AUDIT.md">审计全文</a>。</p>'
    if leg_ik_witness:
        if leg_ik_witness['simulation_calls']!=0 or leg_ik_witness['run_invocations']!=0:
            raise RuntimeError('Preparation witness cannot be presented as a simulation result')
        component_text+='<h4>腿部位置 IK 诊断</h4><p>这是经典两连杆逆运动学的参考表示诊断，不是第四个创新点。原独立见证保留为 '+html.escape(leg_ik_witness['status'])+'：CPU 检查退出为零，但日志距离的逐位相等复核失败。后续独立诊断确认原文档的几何约束均满足，两套参考数组逐项一致，日志距离最大差约 7×10⁻¹⁷ 米；原 FAIL 未覆盖或重写。<a href="../support_leg_ik_20261012/DOCUMENT_WITNESS.md">原见证</a> · <a href="../support_leg_ik_20261012/SCALAR_CONTRACT_DIAGNOSIS.md">独立补充诊断</a>。原目标、前缀与末帧保持不等于动力学可执行，也不绕过 C2/C3 启动条件。</p>'
        if leg_ik_continuation:
            component_text+='<p>初次续跑见证保留为 '+html.escape(leg_ik_continuation['status'])+'：核验额外要求整份历史账本等于后续追加两次完成实验的账本。独立补充核对精确复现原基线收费行及历史哈希，并确认新增收费对应已完成的两组父级实验。<a href="../support_leg_ik_20261012/CONTINUATION_DOCUMENT_WITNESS.md">原续跑见证</a> · <a href="../support_leg_ik_20261012/CONTINUATION_CONTRACT_ADDENDUM.md">独立账本核对</a>。冻结参考、模型、评分和分析未改变。</p>'
        if leg_ik_adjudicated_witness:
            component_text+='<p>分别绑定原失败和独立补充核验的新入口，其独立文档见证为 '+html.escape(leg_ik_adjudicated_witness['status'])+'。<a href="../support_leg_ik_20261012/ADJUDICATED_DOCUMENT_WITNESS.md">最终入口见证</a>。准备通过不代表执行成功或创新有效。</p>'
        if leg_ik_analysis:
            leg_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (arm,r['inputs'],r['counts']['completed'],r['counts']['condition_pass'],r['counts']['physical_pass'],r['counts']['joint_pass'],round(r['means']['surface_slip_proxy_fraction'],5)))+'</tr>' for arm,r in leg_ik_analysis['reports'].items())
            component_text+='<p>24 个内部验证来源、3 种偏移、种子 11，各组 72 输入，全数保留。原参考复用历史回放，两修正组新增回放；未新增生成或训练。</p><table><tr><th>参考</th><th>输入</th><th>完成</th><th>条件合格</th><th>物理合格</th><th>联合合格</th><th>滑移代理均值</th></tr>'+leg_rows+'</table><p>冻结原始数组分析决策：'+html.escape(leg_ik_analysis['decision'])+'。仅支持此参数化的一种子工程诊断；未检验 G1、下游训练、语义与多样性，也未确认创新。<a href="../support_leg_ik_20261012/'+leg_ik_analysis_paths[-1].name+'">原始分析</a>。</p>'
        elif leg_ik_summary:
            component_text+='<p>两次仿真回放已有完成记录，尚待原始数组分析与审计，不作有效性结论。</p>'
        else:
            component_text+='<p>尚无两组仿真完成记录，不作有效性结论。</p>'
        if leg_ik_post_audit:
            component_text+='<p>事后独立审计：'+html.escape(str(leg_ik_post_audit.get('overall_verdict',leg_ik_post_audit.get('verdict','Not verified'))))+'（同模型家族，暂定）。<a href="../support_leg_ik_20261012/POST_RUN_AUDIT.md">审计全文与适用边界</a>。</p>'
        else:
            component_text+='<p>事后独立审计尚未同步，以上为冻结分析器重算结果。</p>'
        if leg_ik_velocity_paths:
            component_text+='<p>另保留足部材料点速度的只读分解：质心平移项与角速度项均有贡献，两种位置修正没有消除滑移代理。分解采用支撑观测加权，不能替代上表逐输入均值，也不能证明单独的因果来源。<a href="../support_leg_ik_20261012/'+leg_ik_velocity_paths[-1].name+'">速度分解记录</a>。</p>'
        component_text+='<p>已核对实际部署的冻结接口：生成参考主要向策略提供身体位置和线速度；不能只新增足部四元数或接触力字段就声称策略会执行它们。位置IK不等于接触动力学修正。<a href="../../docs/remaining_validation_decision_20261012.md">后续验证边界及源代码依据</a>。</p>'
        if child_binding_check:
            component_text+='<p>针对子进程输入绑定缺口，另准备实际加载参考与源码核验入口。本机及服务器的明确合成数组检查通过，并拒绝错误哈希、源码改变和参考内容改变。仅为功能检查，未运行新仿真，未获得独立GPU入口见证，不追溯认证旧实验。<a href="../../docs/child_input_binding_20261012.md">修复边界</a> · <a href="../child_input_binding_20261012/cpu_checks.json">检查记录</a>。</p>'
    if paired51_preview:
        assert paired51_preview['source_sha256']==hashlib.sha256((ROOT/'scripts/preview_paired_feedback_v51_20261012.py').read_bytes()).hexdigest()
        component_text+='<h4>固定样本的状态可视化</h4><p>每类第一个零偏移来源、种子11，样本选择不依赖得分。蓝色为保存的实际人体仿真状态，灰色为所选参考；金色叉号是原始目标。这是骨架回放，不是仿真器相机录像、机器人模型录像或实机测试。</p>'
        for example in paired51_preview['examples']:
            video='../paired_feedback_validation_v51_20261012/previews/'+example['video']
            component_text+='<p>'+html.escape(example['family'])+'</p><video controls preload="metadata" style="max-width:100%;width:800px" src="'+video+'"></video>'
if controller_state_original_failure:
    component_text+='<h4>CLoSD官方训练状态敏感性诊断</h4><p>第一入口在仿真前因缺少上游版本字段退出，独立FAIL及空raw目录保留；第二入口单独修复绑定字段，不改变参考、评分或模型。三个已发布检查点属于同一个CLoSD算法的不同训练状态，不能作为三个独立算法或新创新点。<a href="../controller_state_diagnosis_20261012/DOCUMENT_WITNESS.md">原失败</a> · <a href="../../docs/controller_state_diagnosis_v2_20261012.md">新诊断协议</a>。</p>'
if controller_state_analysis:
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (state,72,r['counts']['completed'],r['counts']['condition_pass'],r['counts']['physical_pass'],r['counts']['joint_pass'],round(r['means']['surface_slip_proxy_fraction'],5)))+'</tr>'
        for state,r in controller_state_analysis['reports'].items())
    component_text+='<p>同一24个内部验证来源、三种偏移、种子11；同一固定参考，生成采样0次。三组导出的初始状态逐元素一致，实际加载的参考与权重字节已记录。隐藏PhysX状态一致性不因此得到认证。</p><table><tr><th>CLoSD训练状态</th><th>输入</th><th>完成</th><th>条件合格</th><th>物理合格</th><th>联合合格</th><th>滑移代理均值</th></tr>'+rows+'</table><p>此项只诊断控制器训练状态对当前人体仿真代理评分的影响，不证明生成方法创新或机器人下游收益，也不追溯修改C1。<a href="../controller_state_diagnosis_v2_20261012/analysis.json">原始数组分析</a>。</p>'
    if controller_state_audit:
        component_text+='<p>独立事后审计：'+html.escape(str(controller_state_audit.get('overall_verdict',controller_state_audit.get('verdict','Not verified'))))+'，同模型家族且暂定。<a href="../controller_state_diagnosis_v2_20261012/EXPERIMENT_AUDIT.md">审计边界</a>。</p>'
    else:
        component_text+='<p>独立事后审计尚未完成，不作有效性结论。</p>'
resource_text='<p>累计资源快照尚未同步。</p>'
if resource_status:
    resource_text=f'<p>累计保守计费{resource_status["charged_wall_hours_one_gpu_per_job"]:.3f} GPU小时，输出{resource_status["output_bytes"]/1024**3:.3f} GiB（含公开导出重复副本），快照时数据盘可用{resource_status["data_free_bytes"]/1024**3:.2f} GiB。总计72小时、阶段上限、输出15GiB和磁盘至少20GiB的检查结果：{html.escape(resource_status["status"])}。阶段ablation_g1包括人体消融计费，不表示已运行G1；下游训练未运行。</p>'
document=f'''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>条件动作生成：接触测量复核与候选机制</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.75 system-ui;color:#172635}}h1,h2{{color:#153c5a}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #cbd5df;padding:10px;text-align:left}}th{{background:#eaf1f7}}.notice{{padding:16px;background:#fff2d8;border-left:5px solid #b67500}}a{{color:#126496}}@media print{{body{{max-width:none;margin:0}}tr{{break-inside:avoid}}}}</style>
<h1>条件动作生成：测量复核与候选机制</h1><p>2026-10-12 · 冻结生成模型和人体跟踪器 · 仅仿真 · 实验结果暂不写入Overleaf</p>
<div class="notice"><strong>3个候选、1个已完成效力对比、2个尚未完成效力对比、0个确认创新。</strong>{measurement_text}程序修复、传感器记录、文档执行见证均不作为论文创新。</div>
<h2>C1完整对比进度</h2>{formal_text}<h2>失败分量与消融证据</h2>{component_text}<h2>原始MoCap参考执行对照</h2>{mocap_text}<h2>C2重复反馈前提诊断</h2>{repeat_text}<h2>本轮测量证据</h2><table><tr><th>试验</th><th>状态</th><th>实际证据与边界</th></tr>{table}</table>
<p>简单物体测试采用5公斤箱体、平面、已知重力和无摩擦滑动。稳定支撑力接近49.05N，初始腾空接触力接近零。约束力传感器重复一致，但关节系统中的传感器还可能包含内部约束反力，因此不能将箱体结果推广为人体接触测量通过。</p>
<p>V11只证明固定新设置内的重复性，不能将其解释为行为不变的修复。初始七项状态与旧设置一致，第一步微小数值差异随后由闭环放大。原阈值保持不变；小样本穿透12/12通过、连续性7/12通过、脚滑代理0/12通过，联合0/12。</p><h2>全来源验证诊断</h2>{validation_text}<h2>速度计算修正与验证边界</h2><p>V19确认原计算将质心线速度当作刚体原点线速度，脚底材料点速度需要扣除旋转后的质心偏移。已单独修正并完成已知静止/滑动/腾空箱体和旋转材料点校验。V21的12条人体输入仍未达到原物理判据，因此计算错误不能解释全部质量问题。历史V12的432次评分没有覆盖正确速度约定，暂不作为正式有效性结论；V24修正后的结果见上表。所有阈值与原始目标保持不变。</p><h2>固定参考回放</h2>{replay_text}<h2>C1的既有初步结果（旧导出设置）</h2><table><tr><th>方法</th><th>输入数</th><th>平均目标误差(cm)</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th></tr>{probe_table}</table>
<p>仅4个动作来源、3种偏移、1个生成种子，每臂最多3候选。条件改善是探索性信号；物理代理尚不可靠，联合结果不能证明方法有效或普遍无效。原始目标、失败和阈值均保留。并非原计划432输入的正式验证。</p>
<h2>三个候选机制</h2><ol><li><strong>C1：有界条件误差补偿。</strong>按实际执行误差有限调整下一轮生成条件。完整3种子对比已结束，未通过联合效力门；新颖性及下游效用均未成立。</li><li><strong>C2：可靠反馈与质量守护。</strong>估计反馈可靠性，拒绝使条件或物理质量恶化的候选。核心、仿真入口和配对分析已实现，数值检查通过；尚未完成仿真效力试验。重复执行计入三次预算；C1联合合格全为零，先诊断跟踪／表示／物理代理，不自动越过预先记录的前提。</li><li><strong>C3：按修复价值分配预算。</strong>区分参考缺陷与控制器困难，按单位成本的预计合格增益与来源覆盖分配候选。预算分配核心已实现并通过合成数值样例检查，尚未完成验证集收益校准与真实仿真效力对比。总预算若为3N且每输入上限3次，各方法调用分配相同；后续需单独冻结稀缺等预算协议，不能用已评分候选库代替实际获取成本。</li></ol>
<p>物理精化、执行反馈、闭环数据扩增已有前人路线。以上具体机制还需最近邻方法核查和等预算消融；不能直接写成论文贡献。</p>
<h2>进入正式验证的条件</h2><p>接触和时间对齐测量校验通过后，补齐独立来源、3生成种子和缺失消融。联合合格率相对最强等预算基线至少提高10个百分点，来源聚类95%置信区间排除零，并满足语义与多样性护栏，才能进入G1验证。下游收益还需相同训练预算、3训练种子与独立测试集。</p>
<h2>资源边界</h2>{resource_text}<h2>记录与复现</h2><ul><li><a href="../contact_force_fixture_20261011/repeat_comparison.json">物理箱体重复比较</a> · <a href="../contact_force_fixture_20261011/DOCUMENT_WITNESS.md">独立文档见证</a></li><li><a href="../contact_sampling_v8_20261011/sampling_summary.json">V8结果</a> · <a href="../contact_sampling_v8_20261011/EXPERIMENT_AUDIT.md">V8独立审计</a></li><li><a href="../../docs/candidate_mechanisms_20261011.md">候选机制与验证边界</a></li><li><a href="status.json">当前状态及证据哈希</a> · <a href="https://github.com/Pendragon08/conditional-human-motion-generation">代码与实验记录仓库</a></li></ul></html>'''
(OUT/'measurement_review.html').write_text(document,encoding='utf8')
print(OUT/'measurement_review.html')
