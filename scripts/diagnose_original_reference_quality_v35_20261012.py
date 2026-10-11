"""Read-only original-reference condition/continuity attribution.

The applied dataset reference is not physical ground truth. Reference jerk is
a kinematic diagnostic, not a new physical acceptance rule. No new simulation,
generation, threshold edits, sample exclusion, or controller changes occur.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from diagnose_condition_feedback_components_v32_20261012 import flags
from physics_guided_metric_v23_20261012 import heading, wrap_angle

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/mocap_reference_control_v33_20261012'
OUT = ROOT / 'results/original_reference_attribution_v35_20261012'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jerk95(positions, dt):
    value = positions
    for _ in range(3):
        value = np.diff(value, axis=0) / dt
    return float(np.quantile(np.linalg.norm(value, axis=-1), .95)) if len(value) else None


def endpoint_time(positions, joint, target, context_steps, dt):
    distances = np.linalg.norm(positions[:, joint] - target, axis=-1)
    eligible = distances[context_steps:]
    near = np.flatnonzero(eligible <= eligible.min() + .001) + context_steps
    best = near[np.argmin(np.abs(near - (len(positions) - 1)))]
    return float(abs(best - (len(positions) - 1)) * dt)


def main(check=False):
    if check:
        t = np.arange(20, dtype=float) * .05
        constant = np.ones((20, 2, 3))
        assert jerk95(constant, .05) == 0
        cubic = np.zeros_like(constant)
        cubic[:, :, 0] = (t ** 3)[:, None]
        assert abs(jerk95(cubic, .05) - 6) < 1e-8
        assert endpoint_time(constant, 0, constant[-1, 0], 4, .05) == 0
        print('ORIGINAL_REFERENCE_ATTRIBUTION_NUMERIC_PASS')
        return
    assert np.__version__ == '1.23.5', 'Use the frozen recording numerical runtime'
    assert read(BASE / 'queue_state.json')['status'] == 'COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC'
    summary = read(BASE / 'diagnostic_summary.json')
    thresholds = read(BASE / 'frozen_thresholds.json')['thresholds']
    rows, bindings = [], {}
    for split, label in [('validation', 'val_mocap_replay_s11'), ('test', 'test_mocap_replay_s11')]:
        path = BASE / 'raw' / label
        metadata, measured, receipt = (read(path / name) for name in
                                     ('metadata.json', 'metrics.json', 'receipt.json'))
        assert receipt['status'] == 'complete'
        assert metadata['generation_batch_calls'] == 0
        for name, digest in summary[split]['output_hashes'].items():
            assert sha(path / name) == digest
        assert sha(path / 'dataset_reference.npz') == metadata['dataset_reference_sha256']
        bindings[label] = {name: sha(path / name) for name in
                          ('metadata.json', 'metrics.json', 'receipt.json',
                           'dataset_reference.npz', 'trajectories.npz')}
        with np.load(path / 'dataset_reference.npz', allow_pickle=False) as original, \
                np.load(path / 'trajectories.npz', allow_pickle=False) as replay:
            reference, valid = original['reference'], replay['valid']
            canonical = (reference - reference[:, :, 0:1]) + reference[:, :, 0:1]
            assert np.allclose(replay['reference'][valid], canonical[valid], atol=1e-12, rtol=0)
            assert np.array_equal(replay['reference_frame'], np.arange(120))
            assert np.array_equal(original['original_goal'], np.asarray(metadata['original_goal'], np.float32))
            for index, (case, metric) in enumerate(zip(metadata['cases'], measured)):
                assert case['case_id'] == metric['case_id']
                state = flags(metric, {'construction_complete': True}, thresholds)
                positions = canonical[:, index]
                target = np.asarray(metadata['original_goal'][index])
                joint = metadata['names'].index(case['goal_joint_body'])
                planned_position = float(np.linalg.norm(positions[-1, joint] - target))
                planned_heading = float(abs(wrap_angle(heading(positions[-1], metadata['names']) - metadata['original_heading'][index])))
                planned_time = endpoint_time(positions, joint, target, metadata['context_steps'], metadata['dt'])
                planned_jerk = jerk95(positions[metadata['context_steps']:], metadata['dt'])
                actual_jerk = jerk95(replay['actual'][valid[:, index], index][min(metadata['context_steps'], int(valid[:, index].sum()) - 1):], metadata['dt'])
                assert actual_jerk == metric['jerk_p95_m_s3']
                rows.append(dict(split=split, case_id=case['case_id'], family=case['family'],
                    offset_m=case['offset_m'], completed=state['replay_complete'],
                    original_reference_endpoint_error_m=planned_position,
                    original_reference_heading_error_rad=planned_heading,
                    original_reference_endpoint_time_error_s=planned_time,
                    original_reference_jerk_p95_m_s3=planned_jerk,
                    executed_jerk_p95_m_s3=actual_jerk,
                    reference_continuity_predicate=(planned_jerk is not None and planned_jerk <= thresholds['continuity_jerk_p95_m_s3']),
                    executed_component_predicates=state))
    groups = []
    for split in ('validation', 'test'):
        for offset in (None, 0.):
            selected = [r for r in rows if r['split'] == split and (offset is None or r['offset_m'] == offset)]
            groups.append(dict(split=split, offset='all' if offset is None else 'zero', attempts=len(selected),
                component_pass_counts={k: sum(r['executed_component_predicates'][k] for r in selected)
                                       for k in selected[0]['executed_component_predicates']},
                reference_continuity_pass=sum(r['reference_continuity_predicate'] for r in selected),
                complete_reference_continuity_pass_but_executed_fail=sum(r['completed'] and r['reference_continuity_predicate'] and not r['executed_component_predicates']['continuity'] for r in selected),
                reference_endpoint_error_max_m=max(r['original_reference_endpoint_error_m'] for r in selected),
                reference_heading_max_rad=max(r['original_reference_heading_error_rad'] for r in selected),
                reference_endpoint_time_max_s=max(r['original_reference_endpoint_time_error_s'] for r in selected),
                failure_signatures=dict(Counter('|'.join(k for k in ('surface_slip_proxy', 'continuity', 'tracking_mean', 'position', 'heading', 'timing', 'replay_complete') if not r['executed_component_predicates'][k]) for r in selected))))
    output = dict(status='COMPLETE_READ_ONLY_ORIGINAL_REFERENCE_ATTRIBUTION', time=time.time(),
        source_sha256=sha(Path(__file__)), metric_source_sha256=sha(ROOT / 'scripts/physics_guided_metric_v23_20261012.py'),
        flags_source_sha256=sha(ROOT / 'scripts/diagnose_condition_feedback_components_v32_20261012.py'),
        thresholds_sha256=sha(BASE / 'frozen_thresholds.json'), input_bindings=bindings,
        summary_sha256=sha(BASE / 'diagnostic_summary.json'), numpy_version=np.__version__,
        simulation_calls=0, generation_calls=0, thresholds_changed=False, weights_updated=False,
        innovation_confirmed=False, original_reference_is_physical_ground_truth=False,
        scope='Dataset-derived reference kinematics versus one fixed tracker; not causal attribution, new efficacy evidence, contact truth or acceptance-threshold revision',
        groups=groups, entries=rows)
    OUT.mkdir(exist_ok=True)
    destination = OUT / ('diagnostic_%d.json' % time.time_ns())
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'groups': groups, 'path': str(destination)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    main(parser.parse_args().self_check)
