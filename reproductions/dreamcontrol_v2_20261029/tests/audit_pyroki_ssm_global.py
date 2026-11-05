"""Audit SSM full-trajectory output and compare it to saved window solutions."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'
BASE = ROOT / 'pyroki_ssm_000008_human_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    baseline = json.loads((BASE / 'report.json').read_text())
    old_audit = json.loads((BASE / 'audit_report.json').read_text())
    assert report['training_ready'] is False and old_audit['training_ready'] is False
    assert sha(SOURCE) == report['source_sha256'] == old_audit['source_sha256']
    assert sha(URDF) == report['urdf_sha256'] == old_audit['urdf_sha256']
    assert sha(HERE / 'run.py') == report['script_sha256']
    assert sha(HERE / 'trajectory.npz') == report['artifact_sha256']
    assert sha(BASE / 'report.json') == old_audit['report_sha256']
    assert sha(BASE / 'windows.jsonl') == baseline['rows_sha256']
    with np.load(SOURCE, allow_pickle=False) as src, np.load(HERE / 'trajectory.npz', allow_pickle=False) as d:
        ix, root, q = d['sample_indices'].copy(), d['root_wxyz_xyz'].copy(), d['joint_pos'].copy()
        names = [str(s) for s in d['joint_names']]
        np.testing.assert_array_equal(ix, np.arange(1, 43))
        np.testing.assert_array_equal(d['source_frames'], src['source_frames'][ix])
        np.testing.assert_array_equal(d['source_times'], src['source_times'][ix])
    assert root.shape == (42, 7) and q.shape == (42, 27)
    assert np.isfinite(root).all() and np.isfinite(q).all()
    assert np.allclose(np.linalg.norm(root[:, :4], axis=1), 1., atol=1e-4, rtol=0)
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        kind = joint.get('type')
        if kind == 'continuous': limits[joint.get('name')] = (-np.inf, np.inf)
        elif kind == 'revolute':
            node = joint.find('limit')
            limits[joint.get('name')] = (float(node.get('lower')), float(node.get('upper')))
    assert len(names) == 27 and set(names) == set(limits)
    lo = np.array([limits[n][0] for n in names]); hi = np.array([limits[n][1] for n in names])
    violation = float(np.maximum(np.maximum(lo - q, q - hi), 0).max())
    rows = [json.loads(s) for s in (BASE / 'windows.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    root_diff, joint_diff, rotation_diff = [], [], []
    for row in rows:
        path = BASE / row['artifact']
        assert sha(path) == row['artifact_sha256']
        with np.load(path, allow_pickle=False) as d:
            wi, wr, wq = d['sample_indices'].copy(), d['root_wxyz_xyz'].copy(), d['joint_pos'].copy()
            assert [str(s) for s in d['joint_names']] == names
        glob = wi - 1
        root_diff.extend(np.linalg.norm(root[glob, 4:] - wr[:, 4:], axis=1).tolist())
        joint_diff.extend(np.max(np.abs(q[glob] - wq), axis=1).tolist())
        dot = np.abs(np.sum(root[glob, :4] * wr[:, :4], axis=1))
        norms = np.linalg.norm(root[glob, :4], axis=1) * np.linalg.norm(wr[:, :4], axis=1)
        rotation_diff.extend((2 * np.arccos(np.clip(dot / norms, 0, 1))).tolist())
    result = {'report_sha256': sha(HERE / 'report.json'), 'artifact_sha256': sha(HERE / 'trajectory.npz'),
              'baseline_report_sha256': sha(BASE / 'report.json'), 'frames_verified': len(ix),
              'max_joint_limit_violation_rad': violation, 'window_frame_comparisons': len(root_diff),
              'global_vs_window_root_m_mean_max': [float(np.mean(root_diff)), float(np.max(root_diff))],
              'global_vs_window_rotation_rad_mean_max': [float(np.mean(rotation_diff)), float(np.max(rotation_diff))],
              'global_vs_window_joint_rad_mean_max': [float(np.mean(joint_diff)), float(np.max(joint_diff))],
              'max_adjacent_root_m': float(np.linalg.norm(np.diff(root[:, 4:], axis=0), axis=1).max()),
              'max_adjacent_joint_rad': float(np.abs(np.diff(q, axis=0)).max()), 'training_ready': False,
              'limitations': 'Same source and old helper; differences are not truth errors. No independent solve, physical, caption, license or training acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
