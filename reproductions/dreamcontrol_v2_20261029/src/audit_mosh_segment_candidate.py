"""Independently recheck crop provenance, selected blends, limits and timing."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN = ROOT / 'mosh_000798_gpu_resume_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031/output'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = json.loads((HERE / 'report.json').read_text())
    h = json.loads((HUMAN / 'human_report.json').read_text())
    assert sha(HERE / 'candidate.npz') == r['output_sha256']
    assert sha(RUN / 'report.json') == r['windows_report_sha256']
    assert sha(RUN / 'audit_report.json') == r['windows_audit_sha256']
    assert sha(RUN / 'windows.jsonl') == r['rows_sha256']
    assert sha(HUMAN / 'human_report.json') == r['human_report_sha256']
    assert sha(URDF) == r['urdf_sha256']
    assert r['source_sha256'] == h['source_sha256'] and r['caption_member_sha256'] == h['caption_member_sha256']
    assert r['original_caption_records'] == h['original_caption_records'] and r['official_split'] == h['official_split']
    rows = [json.loads(x) for x in (RUN / 'windows.jsonl').read_text().splitlines()]
    with np.load(HERE / 'candidate.npz', allow_pickle=False) as d:
        raw, blend, grid, filtered = (d[x].copy() for x in ['raw_frames', 'blended_120hz', 'frames_20hz', 'filtered_20hz'])
        features, names = d['features'].copy(), list(d['joint_names'])
    np.testing.assert_array_equal(raw, np.arange(125, 1015))
    np.testing.assert_array_equal(grid, np.arange(126, 1015, 6))
    assert filtered.shape == (149, 34) and features.shape == (148, 263) and np.isfinite(features).all()
    weights = [1, 2, 3, 4, 3, 2, 1]
    max_error = 0.
    for frame in [126, 582, 1014]:
        values = []
        for row in rows[31:]:
            if frame not in row['sample_indices']:
                continue
            p = RUN / row['artifact']
            assert sha(p) == row['artifact_sha256']
            with np.load(p, allow_pickle=False) as d:
                i = int(np.flatnonzero(d['source_frames'] == frame)[0])
                values.append((weights[i], np.r_[d['root_wxyz_xyz'][i], d['joint_pos'][i]].astype(float)))
        assert values
        base = values[0][1][:4]
        total = np.zeros(34)
        for weight, v in values:
            if np.dot(v[:4], base) < 0:
                v[:4] *= -1
            total += weight * v
        total /= sum(w for w, _ in values)
        total[:4] /= np.linalg.norm(total[:4])
        actual = blend[frame - 125]
        if np.dot(total[:4], actual[:4]) < 0:
            total[:4] *= -1
        max_error = max(max_error, float(np.max(np.abs(total - actual))))
    assert max_error < 1e-6
    limits = {}
    for j in ET.parse(URDF).getroot().findall('joint'):
        if j.get('type') == 'revolute':
            lim = j.find('limit')
            limits[j.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
        elif j.get('type') == 'continuous':
            limits[j.get('name')] = (-float('inf'), float('inf'))
    maximum = 0.
    for pose in filtered:
        for name, value in zip(names, pose[7:]):
            lower, upper = limits[name]
            maximum = max(maximum, lower - value, value - upper)
    assert abs(maximum - r['joint_limit_violation_filtered_rad']) < 1e-12
    out = {'candidate_report_sha256': sha(HERE / 'report.json'), 'output_sha256': sha(HERE / 'candidate.npz'),
           'checked_blend_frames': [126, 582, 1014], 'max_blend_component_error': max_error,
           'filtered_max_joint_limit_violation_rad': maximum,
           'nominal_original_frame_interval': [0, 1014], 'nominal_crop_frame_interval': [126, 1014],
           'nominal_prefix_removed_seconds': 126 / 120, 'caption_accepted': False, 'training_ready': False,
           'limitations': 'Selected blend checks and limits only; no independent FK, FIR, full encoding, source truth or caption semantics.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
