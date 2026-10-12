"""Analyze completed validation only; preserve construction failures in denominator.

No generation, simulation, threshold calibration, test or acceptance claims.
The uncertainty interval resamples original sources (all offsets/seeds together).
"""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paired_feedback_validation_v51_20261012'
ANALYSIS = ROOT / 'results/paired_feedback_analysis_v53_20261012'
bound = {}
provenance_doc = ROOT/'docs/paired_feedback_v51_evaluation_provenance_20261012.md'
bound[str(provenance_doc.relative_to(ROOT))] = hashlib.sha256(provenance_doc.read_bytes()).hexdigest()

def read(path):
    blob = path.read_bytes()
    bound[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)

summary = read(OUT / 'summary.json')
state = read(OUT / 'state.json')
assert state['status'].startswith('COMPLETE_VALIDATION_')
assert summary['status'] == state['status']
assert summary['seeds'] == [11, 12, 13] and summary['inputs_per_method'] == 216
versions = read(OUT / 'versions.json')
for name, digest in versions['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    bound[name] = digest
ledger = read(OUT / 'budget_ledger.json')
assert len(ledger) == 21 and len({x['label'] for x in ledger}) == 21
assert all(x['exit_code'] == 0 for x in ledger)
assert sum(x['seconds'] for x in ledger) <= 3600
thresholds = read(OUT / 'frozen_thresholds.json')['thresholds']
rows = []
for seed in summary['seeds']:
    results = {}
    for arm in ('feedback', 'random'):
        label = '%s_s%d' % (arm, seed)
        replay = OUT / 'raw' / ('val_%s_replay_s%d' % (arm, seed))
        metrics = read(replay / 'metrics.json')
        selected = read(OUT / 'selected' / (label + '.json'))
        metadata = read(replay / 'metadata.json')
        assert len(metrics) == len(selected['choices']) == 72
        assert [x['case_id'] for x in metrics] == [x['case_id'] for x in selected['choices']]
        assert selected['construction_candidates'] == 3
        assert metadata['original_goal_sha256'] == selected['original_goal_sha256']
        counts = {key: sum(bool(x.get(key, False)) for x in metrics) for key in
                  ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')}
        assert counts == summary['reports'][label]['replay']
        results[arm] = metrics
        for metric, choice in zip(metrics, selected['choices']):
            assert 0 <= choice['candidate'] < 3
            record = dict(metric, arm=arm, generation_seed=seed,
                selected_construction_complete=bool(choice['construction_complete']),
                end_to_end_joint_pass=bool(choice['construction_complete'] and metric.get('joint_pass', False)))
            rows.append(record)
    for a, b in zip(results['feedback'], results['random']):
        for key in ('case_id', 'source_id', 'family', 'offset_m', 'original_goal', 'original_heading'):
            assert a[key] == b[key], (seed, key)
    for ordinal in (1, 2):
        a = read(OUT / 'raw' / ('val_feedback_s%d_r%d' % (seed, ordinal)) / 'metadata.json')
        b = read(OUT / 'raw' / ('val_random_s%d_r%d' % (seed, ordinal)) / 'metadata.json')
        assert a['generation_noise_sha256'] == b['generation_noise_sha256']
        assert a['original_goal_sha256'] == b['original_goal_sha256']
sources = sorted({x['source_id'] for x in rows})
assert len(sources) == 24
keys = ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass', 'end_to_end_joint_pass')
totals = {arm: {k: sum(bool(x.get(k, False)) for x in rows if x['arm'] == arm) for k in keys}
          for arm in ('feedback', 'random')}
for arm in totals:
    assert sum(x['arm'] == arm for x in rows) == 216
source_delta = []
for source in sources:
    values = {arm: [float(x['end_to_end_joint_pass']) for x in rows
                    if x['arm'] == arm and x['source_id'] == source] for arm in totals}
    assert all(len(v) == 9 for v in values.values())
    source_delta.append(np.mean(values['feedback']) - np.mean(values['random']))
rng = np.random.RandomState(20261012)
bootstrap = np.asarray(source_delta)[rng.randint(0, 24, size=(10000, 24))].mean(axis=1) * 100
ci = np.percentile(bootstrap, [2.5, 97.5]).tolist()
gain = float(np.mean(source_delta) * 100)
seed_gains = {str(seed): 100 * (sum(x['end_to_end_joint_pass'] for x in rows if x['arm'] == 'feedback' and x['generation_seed'] == seed)
    - sum(x['end_to_end_joint_pass'] for x in rows if x['arm'] == 'random' and x['generation_seed'] == seed)) / 72
    for seed in summary['seeds']}
errors = {}
component_counts = {}
for arm in totals:
    values = [x for x in rows if x['arm'] == arm]
    errors[arm] = {}
    def under(row, key, limit):
        value = row.get(key)
        return isinstance(value, (int, float)) and math.isfinite(value) and value <= limit
    component_counts[arm] = dict(
        position=sum(under(x, 'endpoint_error_m', thresholds['goal_position_m']) for x in values),
        heading=sum(under(x, 'heading_error_rad', thresholds['heading_rad']) for x in values),
        time=sum(under(x, 'endpoint_time_error_s', thresholds['goal_time_s']+1e-6) for x in values),
        surface_slip_proxy=sum(under(x, 'surface_slip_proxy_fraction', thresholds['sliding_fraction']) for x in values),
        penetration=sum(under(x, 'penetration_fraction', thresholds['penetration_fraction']) for x in values),
        continuity=sum(under(x, 'jerk_p95_m_s3', thresholds['continuity_jerk_p95_m_s3']) for x in values),
        construction_complete=sum(x['selected_construction_complete'] for x in values))
    for key in ('endpoint_error_m', 'heading_error_rad', 'endpoint_time_error_s', 'tracking_mean_m',
                'tracking_p95_m', 'surface_slip_proxy_fraction', 'penetration_fraction', 'jerk_p95_m_s3'):
        finite = [x[key] for x in values if isinstance(x.get(key), (int, float)) and math.isfinite(x[key])]
        errors[arm][key] = dict(mean=float(np.mean(finite)) if finite else None,
            recorded_finite=len(finite), missing_or_nonfinite=216-len(finite),
            note='Recorded finite coverage explicitly shown; not used to exclude failures from primary denominator')
result = dict(status='COMPLETE_VALIDATION_ANALYSIS', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_hashes=bound, totals=totals, inputs_per_method=216, original_sources=24, seeds=[11,12,13],
    gain_percentage_points=gain, source_cluster_95_ci_percentage_points=ci, bootstrap_resamples=10000,
    bootstrap_seed=20261012, source_cluster_seed_differences_pp=seed_gains, error_summaries=errors, component_pass_counts=component_counts,
    preliminary_numeric_gate=bool(gain >= 10 and ci[0] > 0 and all(v > 0 for v in seed_gains.values())),
    independent_semantic_diversity_guard='Not completed for this corrected validation epoch',
    acceptance_gate='Not established: validation-only; independent test, guardrails and downstream evidence required',
    evaluation_provenance=dict(
        tracking='synthetic_proxy: fidelity to selected generated/physics-refined simulation trajectories, not dataset or real-robot ground truth',
        condition='Endpoint targets derived from dataset source motion plus predeclared offsets; frozen externally to candidate outputs',
        physical='simulation_only with collision-surface support/slip proxies; no real contact-force or real-slip ground truth',
        joint='Conjunction of fixed external condition checks, simulated physical proxies, completion and synthetic-reference tracking; not certified robot executability'),
    joint_quality_floor=all(x['end_to_end_joint_pass'] == 0 for x in totals.values()),
    interval_scope='Conditional on these 24 validation sources; [0,0] with all-zero outcomes is a floor, not proof of no population effect',
    construction_failure_guard=True, all_failures_retained=True, thresholds_changed=False,
    simulation_calls=0, generation_calls=0, original_scores_changed=False,
    confirmed_innovation=False, g1_or_downstream_run=False, overleaf_results_sync=False,
    own_gpu_seconds=sum(x['seconds'] for x in ledger), time=time.time())
ANALYSIS.mkdir(exist_ok=True)
destination = ANALYSIS / ('analysis_%d.json' % time.time_ns())
destination.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(dict(output=str(destination), totals=totals, gain_pp=gain, source_cluster_ci_pp=ci,
                     joint_quality_floor=result['joint_quality_floor'])), flush=True)
