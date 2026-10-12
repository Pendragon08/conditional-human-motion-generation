"""Read-only paired original-generation versus exact raw-reference replay.

Checks initial states, geometry, applied reference and recorded first action.
No causal assignment, metric change or simulation call is made.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/condition_feedback_formal_v28_20261012'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert np.__version__ == '1.23.5'
    versions = read(BASE / 'versions.json')
    entries, bindings = [], {}
    for seed in (11, 12, 13):
        first = BASE / 'raw' / ('test_planned_s%d_r0' % seed)
        replay = BASE / 'raw' / ('test_eval_raw_s%d' % seed)
        ma, mb = read(first / 'metadata.json'), read(replay / 'metadata.json')
        assert ma['cases'] == mb['cases'] and ma['original_goal'] == mb['original_goal']
        assert ma['original_heading'] == mb['original_heading']
        for path in (first, replay):
            lineage = read(path / 'lineage.json')
            assert lineage['certification'] == 'bound_request_at_execution'
            for name, digest in lineage['outputs'].items():
                assert sha(path / name) == digest
            assert sha(path / 'sensor_diagnostic.npz') == read(path / 'metadata.json')['sensor_diagnostic_sha256']
            bindings[path.name] = {name: sha(path / name) for name in
                ('metadata.json', 'initial_states.npz', 'trajectories.npz', 'sensor_diagnostic.npz')}
        initial = {}
        with np.load(first / 'initial_states.npz', allow_pickle=False) as ia, np.load(replay / 'initial_states.npz', allow_pickle=False) as ib:
            assert set(ia.files) == set(ib.files)
            for key in ia.files:
                assert ia[key].shape == ib[key].shape
                initial[key] = dict(shape=list(ia[key].shape), exactly_equal=bool(np.array_equal(ia[key], ib[key])),
                    maximum_absolute_difference=float(np.max(np.abs(ia[key].astype(float) - ib[key].astype(float)))))
        reference, actual = {}, {}
        valid_counts, paired_actual = {}, {}
        with np.load(first / 'trajectories.npz', allow_pickle=False) as a, np.load(replay / 'trajectories.npz', allow_pickle=False) as b:
            valid = a['valid'] & b['valid']
            wanted = a['reference'] - a['reference'][:, :, 0:1] + a['reference'][:, :, 0:1]
            reference['canonical_max_error_m'] = float(np.max(np.abs(wanted[valid] - b['reference'][valid])))
            assert reference['canonical_max_error_m'] <= 1e-12
            actual = {str(frame): float(np.max(np.abs(a['actual'][frame] - b['actual'][frame])))
                      for frame in (0, 1, 10, 59, 119)}
            for frame in (0, 1, 10, 59, 119):
                keep = valid[frame]
                valid_counts[str(frame)] = int(keep.sum())
                paired_actual[str(frame)] = float(np.max(np.abs(a['actual'][frame, keep] - b['actual'][frame, keep]))) if keep.any() else None
        sensor = {}
        with np.load(first / 'sensor_diagnostic.npz', allow_pickle=False) as sa, np.load(replay / 'sensor_diagnostic.npz', allow_pickle=False) as sb:
            sensor['keys'] = sa.files
            for key in sa.files:
                if key in sb and sa[key].shape == sb[key].shape and sa[key].ndim:
                    sensor[key] = dict(shape=list(sa[key].shape),
                        first_slice_max_difference=float(np.max(np.abs(sa[key][0].astype(float) - sb[key][0].astype(float)))))
        entries.append(dict(seed=seed, attempts=len(ma['cases']), initial_state_fields=initial,
            native_body_properties_equal=ma['native_body_properties'] == mb['native_body_properties'],
            foot_geometry_equal=ma['foot_geometry'] == mb['foot_geometry'],
            reference=reference, actual_pose_max_difference_by_frame_including_invalid=actual,
            common_valid_count_by_frame=valid_counts, valid_pair_actual_pose_max_difference_by_frame=paired_actual,
            sensor=sensor))
    result = dict(status='COMPLETE_RAW_REPLAY_STATE_DIAGNOSTIC', time=time.time(), entries=entries,
        source_sha256=sha(Path(__file__)), versions_sha256=sha(BASE / 'versions.json'), input_bindings=bindings,
        simulation_calls=0, generation_calls=0, historical_scores_changed=False,
        innovation_confirmed=False, overleaf_results_sync=False,
        scope='Original raw references replayed by the existing fixed tracker; numeric/observation diagnosis only, no causal proof')
    out = ROOT / 'results/raw_replay_state_diagnostic_v41_20261012'
    out.mkdir(exist_ok=True)
    target = out / ('diagnostic_%d.json' % time.time_ns())
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'path': str(target), 'entries': entries}, indent=2))


if __name__ == '__main__':
    main()
