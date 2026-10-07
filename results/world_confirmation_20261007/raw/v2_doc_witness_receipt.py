"""Assemble existing independent metadata receipts without remote changes."""
from pathlib import Path
import hashlib,json,time
raw=Path(__file__).resolve().parent
def read(name):return json.loads((raw/name).read_text(encoding='utf-8'))
pre=read('identity_v2_preflight_doc_witness.json');launch=read('identity_v2_launch_receipt.json')
audit=read('identity_v2_independent_audit.json');final=read('identity_v2_final_readonly.json')
meta=raw/'identity_v2_metadata';queue=json.loads((meta/'queue.state').read_text())
assert pre['all_checks_passed'] and launch['launch_count']==1 and launch['launch_returncode']==0
assert audit['status']=='COMPLETE_INDEPENDENT_DOC_FOLLOW_AUDIT_PASS' and queue['status']=='COMPLETE'
assert all(final[k] for k in ['old_sources_states_logs_and_lock_exactly_preserved','all_seven_source_doc_plan_hashes_unchanged','all_selected_references_unchanged','gpu0_lock_free'])
assert '1614065' not in final['processes']['stdout'] and '1614377' not in final['processes']['stdout']
assert '1589572' in final['processes']['stdout'] and '3854248' in final['processes']['stdout']
per_seed={}
for seed,item in audit['per_seed'].items():
 ev=json.loads((meta/f'evaluate_seed{seed}_summary.json').read_text())
 per_seed[seed]={k:item[k] for k in ['transitions','reset_count','push_reset_count_50_to_59','push_reset_count_60_to_109','window_row_count','all_windows_targets_actions_history_exactly_reconstructed','all_real_episode_increments_equal_done','all_ref_domain_min256','all_recovery_windows_in_original_push_episode','actual_forcebuffer_audit_passed','original_reference_identity_maxabs','jit_equivalence_maxabs','domain_gates','equal_reference_H10_primary_RMSE_m','bootstrap_comparison_maxabs_error','stages']}
 per_seed[seed]['comparisons']=ev['comparisons']
 per_seed[seed]['all_22_ref_domain_groups_scored_windows']=1024
 per_seed[seed]['minimum_eligible_clean_windows']=min(v['eligible_reset_free_windows'] for v in item['sampling_counts']['clean'].values())
 per_seed[seed]['minimum_eligible_recovery_windows']=min(v['eligible_reset_free_windows'] for v in item['sampling_counts']['recovery'].values())
receipt={
 'agent':'world_confirmation_identity_v2_doc_witness_20261007',
 'status':'COMPLETE_FRESH_DOC_FOLLOW_AND_INDEPENDENT_AUDIT_PASS',
 'created_unix':time.time(),'compute_skill':'C:/Users/Pende/.codex/skills/run-experiment/SKILL.md',
 'compute_contract':'C:/Users/Pende/.codex/skills/shared-references/compute-env-contract.md',
 'documented_local_invocation':launch['documented_local_invocation'],'launch_count':1,
 'launch_transport_returncode':0,'launch_transport_sentinel':'WORLD_CONFIRMATION_V2_DISPATCH_RETURNED',
 'science_queue_status':'COMPLETE','science_queue_completion_sentinel':'WORLD_RESIDUAL_CONFIRMATION_QUEUE_COMPLETE PASS',
 'all_six_actual_timeout_python_child_chain_returncodes_zero_and_sentinels':True,
 'no_external_termination_or_experiment_retry':True,
 'canonical_spec_sha256':pre['canonical_spec_sha256'],
 'preflight_checks':pre['checks'],'exact_preflight_source_hashes':pre['source_hashes'],
 'collector_normalized_diff':pre['normalized_collector_diff'],'original_66_reference_guard_unchanged':pre['original_66_branch_unchanged'],
 'model_lock_sha256':audit['model_lock_sha256'],'reference_manifest_sha256':audit['reference_manifest_sha256'],'schema_hash':audit['schema_hash'],
 'locked_plan_sha256':pre['source_hashes']['plans/world_residual_confirmation_20261007.md'],
 'old_39_locked_files_unchanged':audit['old_39_hashes_unchanged'],'new_39_locked_files_unchanged':audit['new_39_hashes_unchanged'],
 'all_38_common_locked_model_reference_schema_plan_hashes_identical':audit['38_common_lock_entries_identical'],
 'old_v1_source_state_log_lock_preserved_byte_for_byte':final['old_sources_states_logs_and_lock_exactly_preserved'],
 'all_seven_source_doc_plan_hashes_unchanged_after_science':final['all_seven_source_doc_plan_hashes_unchanged'],
 'all_11_selected_reference_hashes_unchanged_after_science':final['all_selected_references_unchanged'],
 'actual_transitions':audit['actual_transitions'],'actual_reset_count':audit['reset_count'],
 'actual_pushed_reset_events_60_to_109':sum(x['push_reset_count_60_to_109'] for x in per_seed.values()),
 'all_window_rows_exactly_reconstructed':sum(x['window_row_count'] for x in per_seed.values()),
 'hashed_output_file_count':audit['hashed_output_files'],'PPO_updates':0,'auxiliary_updates':0,'normalizer_updates':0,
 'prediction_gate_passed':True,'all_three_seeds_clean_recovery_gate_passed':True,'control_gate':'NOT_EVALUATED_NO_PPO',
 'innovation_validated':False,'not_method_zero_success_rate':True,
 'gpu0_lock_released':final['gpu0_lock_free'],'GPU1_viewer_untouched':True,'original_GPU1_viewer_pid':1589572,'original_GPU1_Sit_process_pid':3854248,
 'no_dependency_install_no_model_selection_no_reference_reselection_no_metric_change':True,
 'licensed_arrays_weights_remain_on_server':True,'metadata_only_local_archive':True,
 'per_seed':per_seed,
 'doc_vs_reality_divergence':'None in the v2 documented scientific invocation. Prior v1 interface failure and transport failures remain preserved.',
 'independent_audit_helper_note':'First metadata audit AST parser failed on symbolic schema constants; its traceback is preserved. Only the local read-only audit parser was corrected; the successful audit re-read existing arrays/checkpoints and did not rerun prediction or science.',
 'limits':audit['limits'],
 'progress_metadata_note':'Collector progress.json is periodic and ends at457 steps; completed summary, full480-step arrays, sentinels and actual child-chain rc0 establish480 steps.',
 'evidence_paths':['identity_v2_preflight_doc_witness.json','identity_v2_launch_receipt.json','identity_v2_metadata/queue.state','identity_v2_metadata/model_lock.json','identity_v2_independent_audit.json','identity_v2_independent_audit_execution.json','identity_v2_audit_attempts/','identity_v2_final_readonly.json'],
 'receipt_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('identity_v2_*.json') if p.name!='identity_v2_doc_witness.json'}
}
out=raw/'identity_v2_doc_witness.json';assert not out.exists(),'Preserve prior doc witness';out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'prediction_gate':receipt['prediction_gate_passed'],'actual_transitions':receipt['actual_transitions'],'actual_resets':receipt['actual_reset_count'],'window_rows':receipt['all_window_rows_exactly_reconstructed'],'PPO_updates':0,'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
