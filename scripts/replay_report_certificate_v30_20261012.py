"""Check published diagnostic artifacts before rendering interface-pass text.

This is an export/report certificate, not a new simulation or binary-array
audit. It does not establish physical quality, true contact, or C1 efficacy.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certify(root, replay):
    root = Path(root)
    checks = {}
    def check(name, predicate):
        checks[name] = bool(predicate)
    def load(path):
        if not path.is_file():
            raise ValueError('Missing published evidence: '+str(path))
        return json.loads(path.read_text(encoding='utf8'))
    try:
        check('exact_terminal_status', isinstance(replay, dict) and replay.get('status') == 'COMPLETE_FIXED_REFERENCE_CALIBRATION')
        if not checks['exact_terminal_status']:
            return dict(status='INCOMPLETE', checks=checks, scope='published_interface_evidence_only')
        check('diagnostic_scope', replay.get('diagnostic_only') is True and replay.get('innovation_confirmed') is False and replay.get('overleaf_results_sync') is False)
        check('source_and_seed_scope', replay.get('source_count') == 24 and replay.get('generation_seeds') == [11])
        entries = replay.get('entries', [])
        check('two_distinct_reference_arms', len(entries) == 2 and {r.get('reference') for r in entries} == {'raw','refined'})
        config = load(root/'configs/physics_guided_20261011.json')
        directory = root/'results/fixed_reference_com_v27_20261012'
        thresholds = load(directory/'frozen_thresholds.json')
        check('unchanged_numeric_thresholds', thresholds['thresholds'] == config['base_thresholds'])
        # The embedded digest hashes sorted parsed JSON, not raw file bytes.
        canonical = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        check('embedded_config_hash_is_sorted_json', thresholds['config_sha256'] == canonical)
        for row in entries:
            ref = row['reference']
            prefix = ref+':'
            path = directory/'public/raw'/('val_'+ref+'_replay_s11')
            meta = load(path/'metadata.json')
            metrics = load(path/'metrics.json')
            receipt = load(path/'receipt.json')
            lineage = load(path/'lineage.json')
            selected = load(directory/'selected'/(ref+'_reference.json'))
            selected_lineage = load(directory/'selected'/(ref+'_reference.lineage.json'))
            check(prefix+'72_retained_inputs', row['attempted'] == 72 and len(metrics) == len(meta['cases']) == receipt['cases'] == len(selected['choices']) == 72)
            check(prefix+'complete_receipt', receipt['status'] == 'complete')
            check(prefix+'metadata_digest', sha(path/'metadata.json') == row['metadata_sha256'] == lineage['outputs']['metadata.json'])
            check(prefix+'metric_digest', sha(path/'metrics.json') == row['metrics_sha256'] == lineage['outputs']['metrics.json'])
            check(prefix+'binary_digest_binding', receipt['trajectories_sha256'] == lineage['outputs']['trajectories.npz'])
            check(prefix+'replay_input_binding', lineage['request']['replay_motion'] == selected_lineage['selected_motion'])
            check(prefix+'request_config_raw_file_hash', lineage['request']['config'] == sha(root/'configs/physics_guided_20261011.json'))
            check(prefix+'request_threshold_binding', lineage['request']['thresholds'] == sha(directory/'frozen_thresholds.json'))
            check(prefix+'selection_digest', sha(directory/'selected'/(ref+'_reference.json')) == selected_lineage['selection_json'])
            check(prefix+'same_goal', meta['original_goal_sha256'] == selected['original_goal_sha256'])
            check(prefix+'case_order', [r['case_id'] for r in metrics] == [r['case_id'] for r in selected['choices']] == [r['case_id'] for r in meta['cases']])
            check(prefix+'completion_fence', meta['completion_fence_calls'] >= 120 and 'fetch_results(True)' in meta['completion_fence_protocol'])
            check(prefix+'reset_guard', meta['initial_reset_gate_pass'] and all(v == 0 for v in meta['manual_reset_vs_first_action_state_max_difference'].values()))
            check(prefix+'native_export_configuration', not meta['sim_params']['use_gpu_pipeline'] and meta['sim_params']['physx_use_gpu'])
            check(prefix+'no_regeneration', meta['generation_batch_calls'] == 0)
            for summary_key, metric_key in (('completed','completed'),('condition_pass','condition_pass'),('physical_pass','physical_pass'),('tracking_pass','tracking_pass'),('joint_pass','joint_pass')):
                check(prefix+summary_key, sum(bool(r.get(metric_key,False)) for r in metrics) == row[summary_key])
            check(prefix+'construction_failures_retained', sum(not c['construction_complete'] for c in selected['choices']) == row['construction_terminated'])
            check(prefix+'combined_success_denominator', sum(bool(m.get('joint_pass',False) and c['construction_complete']) for m,c in zip(metrics,selected['choices'])) == row['combined_construction_replay_joint_pass'])
        check('quality_floor_is_not_pass', replay['full_physical_quality_floor'] == all(r['physical_pass'] == 0 for r in entries))
    except (KeyError, TypeError, ValueError, OSError) as error:
        return dict(status='FAIL', checks=checks, error=str(error), scope='published_interface_evidence_only')
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                scope='published_interface_evidence_only', simulation_reexecuted=False,
                binary_arrays_independently_audited_here=False, innovation_confirmed=False)
