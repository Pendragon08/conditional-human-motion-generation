"""Independently check a synthetic plane and the real body-marker rejection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ground_calibration import fit_ground_surface

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    probe=json.loads((HERE/'probe.json').read_text())
    row=next(r for r in map(json.loads,ALIGN.read_text().splitlines()) if r['motion_id']=='011120' and not r['mirrored'])
    receipt=Path(row['source_receipt']); source=Path(json.loads(receipt.read_text())['smplx_file'])
    assert sha(receipt)==probe['source_receipt_sha256'] and sha(source)==probe['source_sha256']
    with np.load(source,allow_pickle=False) as f:
        labels=[str(x) for x in f['labels']];shape=list(f['markers'].shape)
    assert shape==probe['source_marker_shape'] and all(labels.count(x)==1 for x in probe['foot_body_labels'])
    points=np.array([[[0.,0.,.3],[1.,0.,.4],[0.,1.,.5]],[[0.,0.,.3],[1.,0.,.4],[0.,1.,.5]]])
    up=np.array([0.,0.,1.]); rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    transform=np.eye(4);transform[:3,:3]=rot;transform[:3,3]=[1.,2.,3.]
    out=fit_ground_surface(points,['p0','p1','p2'],observation_kind='fixed_ground_surface_points',
         source_to_robot=transform,source_up=up,stationary_tolerance_m=1e-8,plane_tolerance_m=1e-8,min_normal_alignment=.95)
    normal=np.cross(points[0,1]-points[0,0],points[0,2]-points[0,0]);normal/=np.linalg.norm(normal)
    if normal@up<0:normal=-normal
    expected=rot@normal; origin=rot@points[0,0]+transform[:3,3]
    error=float(np.max(np.abs(expected-out['normal_robot'])))
    assert error<1e-12 and abs(float(-expected@origin)-out['offset_robot'])<1e-12
    assert probe['calibration_status']=='unavailable_body_markers_only' and probe['training_ready'] is False
    assert 'Ran 5 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    result={'probe_sha256':sha(HERE/'probe.json'),'module_sha256':sha(HERE/'ground_calibration.py'),
            'tests_sha256':sha(HERE/'tests.log'),'synthetic_normal_max_error':error,'training_ready':False,
            'limitations':'Synthetic plane check and source labels only; no measured ground, dynamics, calibration provenance or training acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
