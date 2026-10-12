"""Compare event meanings, without changing any historical acceptance scores.

Closest approach, first threshold entry and deadline occupancy are different
events. Neither this diagnostic nor its fixtures select a new acceptance rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/mocap_reference_control_v33_20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def events(distances, context, dt, position_tolerance, expected):
    distances = np.asarray(distances)
    final_frame = expected - 1
    d = distances[context:]
    if not len(d):
        return dict(has_generated_part=False, closest_approach_frame=None,
            frozen_closest_approach_error_s=float(expected * dt),
            first_entry_frame=None, first_entry_error_s=None, deadline_observed=False,
            deadline_within_position_tolerance=False,
            final_distance_m=float(distances[-1]) if len(distances) else None,
            minimum_distance_m=None)
    near = np.flatnonzero(d <= d.min() + .001) + context
    closest = int(near[np.argmin(abs(near - final_frame))])
    entered = np.flatnonzero(d <= position_tolerance) + context
    first = int(entered[0]) if len(entered) else None
    return dict(has_generated_part=True, closest_approach_frame=closest,
        frozen_closest_approach_error_s=float(abs(closest - final_frame) * dt),
        first_entry_frame=first,
        first_entry_error_s=None if first is None else float(abs(first - final_frame) * dt),
        deadline_observed=len(distances) == expected,
        deadline_within_position_tolerance=bool(len(distances) == expected and distances[-1] <= position_tolerance),
        final_distance_m=float(distances[-1]), minimum_distance_m=float(d.min()))


def self_check():
    # Reaching closer earlier and ending within tolerance is not lateness.
    row = events([.4, .3, .02, .04, .08], 1, .1, .1, 5)
    assert row['closest_approach_frame'] == row['first_entry_frame'] == 2
    assert row['frozen_closest_approach_error_s'] == .2
    assert row['deadline_within_position_tolerance']
    # A constant pose has earliest entry early but closest tie at the deadline.
    row = events([.02] * 5, 1, .1, .1, 5)
    assert row['closest_approach_frame'] == 4 and row['first_entry_frame'] == 1
    assert row['frozen_closest_approach_error_s'] == 0
    assert row['first_entry_error_s'] > 0
    row = events([.4, .3], 1, .1, .1, 5)
    assert not row['deadline_observed'] and row['first_entry_frame'] is None
    row = events([.4], 2, .1, .1, 5)
    assert not row['has_generated_part'] and row['frozen_closest_approach_error_s'] == .5
    print('ENDPOINT_EVENT_DIAGNOSTIC_FIXTURES_PASS')


def main():
    assert np.__version__ == '1.23.5'
    certificates = sorted((ROOT / 'results/mocap_cache_binding_v36_20261012').glob('certificate_*.json'))
    assert certificates, 'Run strict cache binding first'
    certificate = read(certificates[-1])
    assert certificate['status'] == 'PASS'
    # A certificate is not a durable exemption: verify every bound current file.
    for name, digest in certificate['input_bindings'].items():
        assert sha(Path(name)) == digest, name
    thresholds = read(BASE / 'frozen_thresholds.json')['thresholds']
    rows = []
    for split, label in (('validation', 'val_mocap_replay_s11'), ('test', 'test_mocap_replay_s11')):
        path = BASE / 'raw' / label
        meta = read(path / 'metadata.json')
        stored = read(path / 'metrics.json')
        with np.load(path / 'trajectories.npz', allow_pickle=False) as arrays:
            for index, (case, score) in enumerate(zip(meta['cases'], stored)):
                valid = arrays['valid'][:, index]
                joint = meta['names'].index(case['goal_joint_body'])
                pos = arrays['actual'][valid, index, joint]
                distances = np.linalg.norm(pos - np.asarray(meta['original_goal'][index]), axis=-1)
                event = events(distances, meta['context_steps'], meta['dt'],
                               thresholds['goal_position_m'], meta['episode_steps'])
                assert event['frozen_closest_approach_error_s'] == score['endpoint_time_error_s']
                rows.append(dict(split=split, case_id=case['case_id'], offset_m=case['offset_m'],
                    family=case['family'], events=event, stored_condition_pass=score['condition_pass'],
                    stored_physical_pass=score['physical_pass'], stored_tracking_pass=score['tracking_pass'],
                    stored_joint_pass=score['joint_pass'], completed=score['completed']))
    groups = []
    for split in ('validation', 'test'):
        for offset in (None, 0.):
            selected = [row for row in rows if row['split'] == split and (offset is None or row['offset_m'] == offset)]
            mismatch = [row for row in selected if row['events']['deadline_within_position_tolerance'] and
                        row['events']['frozen_closest_approach_error_s'] > thresholds['goal_time_s'] + 1e-6]
            groups.append(dict(split=split, offset='all' if offset is None else 'zero', attempts=len(selected),
                deadline_position_satisfied_but_closest_timing_fails=len(mismatch),
                physically_and_tracking_qualified_subset=sum(row['completed'] and row['stored_physical_pass'] and
                    row['stored_tracking_pass'] for row in mismatch),
                mismatch_examples=[row['case_id'] for row in mismatch]))
    result = dict(status='COMPLETE_EVENT_SEMANTICS_DIAGNOSTIC', time=time.time(),
        source_sha256=sha(Path(__file__)), binding_certificate_sha256=sha(certificates[-1]),
        binding_certificate=str(certificates[-1]), groups=groups, entries=rows,
        acceptance_rule_selected=False, historical_scores_changed=False, thresholds_changed=False,
        simulation_calls=0, generation_calls=0, innovation_confirmed=False, overleaf_results_sync=False,
        scope='Distinct event definitions; no reclassification, new efficacy or permission to bypass the frozen gate')
    out = ROOT / 'results/endpoint_event_diagnostic_v37_20261012'
    out.mkdir(exist_ok=True)
    target = out / ('diagnostic_%d.json' % time.time_ns())
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'path': str(target), 'groups': groups}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    if parser.parse_args().self_check:
        self_check()
    else:
        main()
