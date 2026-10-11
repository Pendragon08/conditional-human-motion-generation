"""Read-only contact-repeatability diagnostics; no quality threshold changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_guided_core import rotate_xyzw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_sampling_v5_20261011'


def support(arrays, meta, thresholds):
    states = []
    for name, geometry in meta['foot_geometry'].items():
        index = meta['names'].index(name)
        points = np.asarray(geometry['points'], dtype=float)
        radii = np.asarray(geometry['radii'], dtype=float)
        rotated = rotate_xyzw(arrays['quaternion'][:, :, index, None, :], points[None, None])
        heights = (arrays['actual'][:, :, index, None, :] + rotated)[..., 2] - radii
        grounded = heights.min(axis=-1) < thresholds['surface_near_ground_m']
        states.append(grounded & (arrays['contact_force'][:, :, index, 2] > thresholds['contact_force_n']))
    return np.stack(states, axis=-1)


def compare(left, right, thresholds):
    metadata = [json.loads((path/'metadata.json').read_text()) for path in (left, right)]
    for key in ('original_goal_sha256', 'generation_noise_sha256', 'generation_prefix_sha256'):
        assert metadata[0][key] == metadata[1][key], 'Repeat inputs differ: ' + key
    assert metadata[0]['cases'] == metadata[1]['cases']
    with np.load(left/'trajectories.npz', allow_pickle=False) as a, np.load(right/'trajectories.npz', allow_pickle=False) as b:
        assert np.array_equal(a['valid'], b['valid'])
        valid = a['valid']
        deltas = {key: float(np.max(np.abs(a[key][valid] - b[key][valid]))) for key in
                  ('actual', 'reference', 'quaternion', 'linvel', 'angvel', 'contact_force')}
        flags = [support(archive, meta, thresholds) for archive, meta in zip((a,b), metadata)]
        intersection = int((flags[0] & flags[1])[valid].sum())
        union = int((flags[0] | flags[1])[valid].sum())
        disagreements = int((flags[0] != flags[1])[valid].sum())
        frame_cases = int((flags[0] != flags[1]).any(axis=-1)[valid].sum())
    rows = [json.loads((path/'metrics.json').read_text()) for path in (left,right)]
    return dict(left=str(left), right=str(right), deltas=deltas,
        support_jaccard=intersection/union if union else None,
        support_label_disagreements=disagreements, frame_case_support_disagreements=frame_cases,
        valid_frame_cases=int(valid.sum()),
        max_slip_fraction_difference=max(abs(x['surface_slip_proxy_fraction']-y['surface_slip_proxy_fraction']) for x,y in zip(*rows)),
        classification_flips={key:sum(x[key]!=y[key] for x,y in zip(*rows)) for key in
                              ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
        raw_array_sha256={str(path): hashlib.sha256((path/'trajectories.npz').read_bytes()).hexdigest() for path in (left,right)})


if __name__ == '__main__':
    thresholds = json.loads((OUT/'frozen_thresholds.json').read_text())['thresholds']
    old = ROOT/'results/condition_feedback_probe_20261011/raw/val_native_s11_r0'
    new = ROOT/'results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r0'
    result = dict(unsynchronized_repeat=compare(old,new,thresholds),
        synchronized_repeat=compare(OUT/'raw/repeat_a',OUT/'raw/repeat_b',thresholds),
        thresholds_changed=False, confirmatory=False, actual_contact_point_truth_available=False,
        role='Sampling repeatability diagnostic, not innovation efficacy or force ground truth')
    path=OUT/'repeatability_summary.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
