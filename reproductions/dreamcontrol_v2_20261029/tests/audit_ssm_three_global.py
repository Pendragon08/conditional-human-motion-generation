"""Audit one full-trajectory output from the three-source SSM candidate batch."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path.cwd()
HUMAN = ROOT / 'ssm_three_human_candidates_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    name = HERE.name
    batch = json.loads((HERE.parent / 'manifest.json').read_text())
    rec = next(x for x in batch['records'] if x['id'] == name)
    human_report = json.loads((HUMAN / 'report.json').read_text())
    human_audit = json.loads((HUMAN / 'independent_audit.json').read_text())
    report = json.loads((HERE / 'report.json').read_text())
    source = HUMAN / (name + '_joint_witness.npz')
    assert sha(HUMAN / 'report.json') == batch['human_report_sha256'] == human_audit['witness_report_sha256']
    assert sha(HUMAN / 'independent_audit.json') == batch['human_audit_sha256']
    assert sha(source) == rec['human_artifact_sha256'] == report['source_sha256']
    assert sha(HERE / 'run.py') == rec['script_sha256'] == report['script_sha256']
    assert sha(HERE / 'trajectory.npz') == report['artifact_sha256']
    assert sha(URDF) == report['urdf_sha256'] and report['training_ready'] is False
    with np.load(source, allow_pickle=False) as s, np.load(HERE / 'trajectory.npz', allow_pickle=False) as d:
        indices, root, q = d['sample_indices'].copy(), d['root_wxyz_xyz'].copy(), d['joint_pos'].copy()
        names = [str(x) for x in d['joint_names']]
        np.testing.assert_array_equal(indices, np.arange(1, rec['input_points']))
        np.testing.assert_array_equal(d['source_frames'], s['source_frames'][indices])
        np.testing.assert_array_equal(d['source_times'], s['source_times'][indices])
    assert root.shape == (len(indices), 7) and q.shape == (len(indices), 27)
    assert np.isfinite(root).all() and np.isfinite(q).all()
    assert np.allclose(np.linalg.norm(root[:, :4], axis=1), 1., rtol=0, atol=1e-4)
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'continuous': limits[joint.get('name')] = (-np.inf, np.inf)
        elif joint.get('type') == 'revolute':
            node = joint.find('limit')
            limits[joint.get('name')] = (float(node.get('lower')), float(node.get('upper')))
    assert len(set(names)) == 27 and set(names) == set(limits)
    lo = np.array([limits[n][0] for n in names]); hi = np.array([limits[n][1] for n in names])
    result = {'id': name, 'report_sha256': sha(HERE / 'report.json'), 'source_sha256': sha(source),
              'trajectory_sha256': sha(HERE / 'trajectory.npz'), 'frames_verified': len(indices),
              'max_joint_limit_violation_rad': float(np.maximum(np.maximum(lo-q, q-hi), 0).max()),
              'max_adjacent_root_m': float(np.linalg.norm(np.diff(root[:, 4:], axis=0), axis=1).max()),
              'max_adjacent_joint_rad': float(np.abs(np.diff(q, axis=0)).max()),
              'training_ready': False,
              'limitations': 'Recorded output audit; no independent solve, caption semantics, model identity, license or physical acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
