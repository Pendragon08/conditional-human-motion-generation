"""Check saved SSM PyRoki windows against source time, URDF limits and hashes.

This does not validate the optimizer, human model, contact, or dynamics.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    assert report['training_ready'] is False
    assert sha(SOURCE) == report['source_sha256']
    assert sha(URDF) == report['urdf_sha256']
    assert sha(HERE / 'run.py') == report['script_sha256']
    assert sha(HERE / 'windows.jsonl') == report['rows_sha256']
    rows = [json.loads(s) for s in (HERE / 'windows.jsonl').read_text().splitlines()]
    assert len(rows) == report['window_count'] == 10
    with np.load(SOURCE, allow_pickle=False) as data:
        source_frames = data['source_frames'].copy()
        source_times = data['source_times'].copy()
    joints = {}
    for node in ET.parse(URDF).getroot().findall('joint'):
        if node.get('type') in ('revolute', 'continuous'):
            lim = node.find('limit')
            if node.get('type') == 'continuous':
                joints[node.get('name')] = (-np.inf, np.inf)
            else:
                joints[node.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
    covered, max_violation, max_root, max_quat, max_joint = set(), 0., 0., 0., 0.
    previous = None
    for k, row in enumerate(rows):
        assert row['window'] == k
        artifact = HERE / row['artifact']
        assert sha(artifact) == row['artifact_sha256']
        with np.load(artifact, allow_pickle=False) as data:
            ix = data['sample_indices'].copy()
            frames = data['source_frames'].copy()
            times = data['source_times'].copy()
            root = data['root_wxyz_xyz'].copy()
            q = data['joint_pos'].copy()
            names = [str(s) for s in data['joint_names']]
        assert ix.tolist() == row['sample_indices'] and len(ix) == 7
        assert np.all(np.diff(ix) == 1) and root.shape == (7, 7) and q.shape == (7, 27)
        np.testing.assert_array_equal(frames, source_frames[ix])
        np.testing.assert_array_equal(times, source_times[ix])
        assert set(names) == set(joints) and len(set(names)) == 27
        assert np.isfinite(root).all() and np.isfinite(q).all()
        assert np.allclose(np.linalg.norm(root[:, :4], axis=1), 1., atol=1e-4, rtol=0)
        lo = np.array([joints[n][0] for n in names])
        hi = np.array([joints[n][1] for n in names])
        max_violation = max(max_violation, float(np.maximum(np.maximum(lo - q, q - hi), 0).max()))
        covered.update(ix.tolist())
        if previous is not None:
            common = sorted(set(ix.tolist()) & set(previous['ix'].tolist()))
            assert common == row['overlap']['sample_indices']
            for idx in common:
                old = np.flatnonzero(previous['ix'] == idx)[0]
                new = np.flatnonzero(ix == idx)[0]
                max_root = max(max_root, float(np.linalg.norm(previous['root'][old, 4:] - root[new, 4:])))
                dot = float(abs(np.dot(previous['root'][old, :4], root[new, :4])))
                norms = float(np.linalg.norm(previous['root'][old, :4]) * np.linalg.norm(root[new, :4]))
                max_quat = max(max_quat, 2 * float(np.arccos(np.clip(dot / norms, 0, 1))))
                max_joint = max(max_joint, float(np.max(np.abs(previous['q'][old] - q[new]))))
        previous = {'ix': ix, 'root': root, 'q': q}
    assert covered == set(range(1, 43))
    result = {'report_sha256': sha(HERE / 'report.json'), 'source_sha256': sha(SOURCE),
              'urdf_sha256': sha(URDF), 'windows_verified': len(rows),
              'covered_indices': [min(covered), max(covered)], 'covered_samples': len(covered),
              'max_joint_limit_violation_rad': max_violation,
              'max_overlap_root_m': max_root, 'max_overlap_rotation_rad': max_quat,
              'max_overlap_joint_rad': max_joint, 'training_ready': False,
              'limitations': 'Checks saved outputs only; no independent optimization, source truth, contact, caption, license or dynamics acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
