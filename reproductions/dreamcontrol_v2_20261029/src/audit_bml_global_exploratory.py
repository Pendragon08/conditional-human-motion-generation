"""Independently check the saved BML candidate trajectory and provenance."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    here = Path.cwd()
    batch = json.loads((here.parent/'manifest.json').read_text())
    item = next(x for x in batch['cases'] if x['label'] == here.name)
    report = json.loads((here/'report.json').read_text())
    assert sha(here/'human.npz') == item['human_sha256'] == report['source_sha256']
    assert sha(here/'run.py') == item['solver_sha256'] == report['script_sha256']
    assert sha(here/'trajectory.npz') == report['artifact_sha256']
    assert sha(URDF) == report['urdf_sha256'] and report['training_ready'] is False
    with np.load(here/'human.npz', allow_pickle=False) as s, np.load(here/'trajectory.npz', allow_pickle=False) as d:
        indices, root, q = d['sample_indices'].copy(), d['root_wxyz_xyz'].copy(), d['joint_pos'].copy()
        names = [str(x) for x in d['joint_names']]
        np.testing.assert_array_equal(indices, np.arange(1,61))
        np.testing.assert_array_equal(d['source_frames'], s['source_frames'][indices])
        np.testing.assert_array_equal(d['source_times'], s['source_times'][indices])
    assert root.shape == (60,7) and q.shape == (60,27)
    assert np.isfinite(root).all() and np.isfinite(q).all()
    assert np.allclose(np.linalg.norm(root[:,:4],axis=1),1,rtol=0,atol=1e-4)
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'continuous': limits[joint.get('name')] = (-np.inf,np.inf)
        elif joint.get('type') == 'revolute':
            lim = joint.find('limit')
            limits[joint.get('name')] = (float(lim.get('lower')),float(lim.get('upper')))
    assert len(set(names)) == 27 and set(names) == set(limits)
    lo=np.array([limits[n][0] for n in names]); hi=np.array([limits[n][1] for n in names])
    out = {'id':here.name,'report_sha256':sha(here/'report.json'), 'trajectory_sha256':sha(here/'trajectory.npz'),
           'input_sha256':sha(here/'human.npz'), 'frames_verified':len(indices),
           'max_joint_limit_violation_rad':float(np.maximum(np.maximum(lo-q,q-hi),0).max()),
           'max_adjacent_root_m':float(np.linalg.norm(np.diff(root[:,4:],axis=0),axis=1).max()),
           'max_adjacent_joint_rad':float(np.abs(np.diff(q,axis=0)).max()), 'training_ready':False,
           'limitations':'Saved-output audit only; no independent solve or proof of physical quality, caption, data identity or license.'}
    with (here/'audit_report_v2.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__ == '__main__':main()
