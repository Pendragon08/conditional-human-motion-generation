"""Independently check candidate provenance, timeline, interpolation and captions.

FK, feature encoding, motion semantics and licensing are not independently proven.
"""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
SOLVE = ROOT / 'pyroki_ssm_000008_global_20261031'
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    assert report['training_ready'] is False and report['motion_id'] == '000008'
    assert sha(HERE / 'export.py') == report['script_sha256']
    assert sha(HERE / 'candidate.npz') == report['output_sha256']
    assert sha(SOLVE / 'trajectory.npz') == report['source_trajectory_sha256']
    assert sha(SOLVE / 'report.json') == report['solve_report_sha256']
    assert sha(SOLVE / 'audit_report.json') == report['solve_audit_sha256']
    assert sha(HUMANML / 'texts.zip') == report['caption_zip_sha256']
    assert sha(HUMANML / 'train.txt') == report['official_split_file_sha256']
    with zipfile.ZipFile(HUMANML / 'texts.zip') as z:
        member = z.read(report['caption_member'])
    assert hashlib.sha256(member).hexdigest() == report['caption_member_sha256']
    lines = member.decode('utf8').splitlines()
    assert len(lines) == len(report['captions']) == 4
    for i, line in enumerate(lines):
        parts = line.split('#')
        assert report['captions'][i] == {'line': i + 1, 'caption': parts[0], 'tag_strings': parts[2:]}
    assert '000008' in (HUMANML / 'train.txt').read_text().splitlines()
    assert all('000008' not in (HUMANML / (part + '.txt')).read_text().splitlines()
               for part in ('val', 'test'))
    with np.load(SOLVE / 'trajectory.npz', allow_pickle=False) as s, np.load(HERE / 'candidate.npz', allow_pickle=False) as d:
        t, grid = s['source_times'].copy(), d['time_grid'].copy()
        source_root, source_q = s['root_wxyz_xyz'].copy(), s['joint_pos'].copy()
        np.testing.assert_array_equal(s['source_times'], d['original_times'])
        np.testing.assert_array_equal(s['source_frames'], d['original_source_frames'])
        np.testing.assert_array_equal(s['sample_indices'], d['original_sample_indices'])
        np.testing.assert_array_equal(s['joint_names'], d['joint_names'])
        target_pos, target_rot, target_q = d['root_position'].copy(), d['root_quaternion_xyzw'].copy(), d['joint_pos'].copy()
        features, points = d['features'].copy(), d['points_z_up_world'].copy()
    assert len(t) == 42 and len(grid) == 41 and features.shape == (40, 263)
    np.testing.assert_allclose(np.diff(grid), 1 / 20, rtol=0, atol=1e-15)
    assert grid[0] == t[0] and grid[-1] <= t[-1] and grid[-1] + .05 > t[-1]
    assert np.isfinite(features).all() and np.isin(features[:, -4:], [0., 1.]).all()
    max_pos = max_joint = max_quat = 0.
    for k, target in enumerate(grid):
        j = min(max(np.searchsorted(t, target, side='right') - 1, 0), len(t) - 2)
        u = (target - t[j]) / (t[j + 1] - t[j])
        expected_pos = (1 - u) * source_root[j, 4:] + u * source_root[j + 1, 4:]
        expected_q = (1 - u) * source_q[j] + u * source_q[j + 1]
        a = source_root[j, [1, 2, 3, 0]].astype(float); a /= np.linalg.norm(a)
        b = source_root[j + 1, [1, 2, 3, 0]].astype(float); b /= np.linalg.norm(b)
        dot = float(np.dot(a, b))
        if dot < 0: b = -b; dot = -dot
        theta = np.arccos(np.clip(dot, -1, 1))
        expected_rot = ((1 - u) * a + u * b) if theta < 1e-7 else (np.sin((1-u)*theta)*a + np.sin(u*theta)*b) / np.sin(theta)
        expected_rot /= np.linalg.norm(expected_rot)
        max_pos = max(max_pos, float(np.max(np.abs(expected_pos - target_pos[k]))))
        max_joint = max(max_joint, float(np.max(np.abs(expected_q - target_q[k]))))
        max_quat = max(max_quat, float(min(np.max(np.abs(expected_rot - target_rot[k])), np.max(np.abs(expected_rot + target_rot[k])))))
    assert max(max_pos, max_joint, max_quat) < 1e-6
    result = {'candidate_report_sha256': sha(HERE / 'report.json'), 'artifact_sha256': sha(HERE / 'candidate.npz'),
        'source_trajectory_sha256': sha(SOLVE / 'trajectory.npz'), 'captions_verified': len(lines),
        'input_points': len(t), 'output_points': len(grid), 'feature_rows': len(features),
        'max_interpolation_component_errors': {'position_m': max_pos, 'joint_rad': max_joint, 'quaternion': max_quat},
        'training_ready': False, 'limitations': 'FK, feature encoding, caption semantics, source identity, physical quality and license not independently established.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
