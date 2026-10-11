"""Calibrate passive sensor diagnostics without replacing frozen metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
from physics_guided_core import rotate_xyzw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_sensor_v10_20261011'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
records=[]
for label in ('repeat_a','repeat_b'):
    path=OUT/'raw'/label
    if not (path/'sensor_diagnostic.npz').exists(): continue
    meta=json.loads((path/'metadata.json').read_text())
    with np.load(path/'trajectories.npz',allow_pickle=False) as a, np.load(path/'sensor_diagnostic.npz',allow_pickle=False) as d:
        heights=[]
        for name in meta['passive_sensor_bodies']:
            body=meta['names'].index(name)
            geom=meta['foot_geometry'][name]
            rotated=rotate_xyzw(a['quaternion'][:,:,body,None,:],np.asarray(geom['points'])[None,None])
            z=a['actual'][:,:,body,None,2]+rotated[...,2]-np.asarray(geom['radii'])[None,None]
            heights.append(z.min(axis=-1))
        heights=np.stack(heights,axis=-1)
        # Clear geometric separation is an airborne diagnostic; it is not a
        # simulator contact witness or a new classification threshold.
        clear=(heights>.05)&a['valid'][:,:,None]
        force=d['sensor'][...,:3]
        per_body=[]
        for i,name in enumerate(meta['passive_sensor_bodies']):
            mask=clear[:,:,i]
            per_body.append(dict(body=name,clear_airborne_observations=int(mask.sum()),
                max_constraint_force_norm_during_clear_separation_N=float(np.linalg.norm(force[:,:,i][mask],axis=-1).max()) if mask.any() else None,
                positive_vertical_sensor_above_10N_when_clearly_separated=int((force[:,:,i,2][mask]>10).sum()),
                vertical_sensor_range_N=[float(force[:,:,i,2].min()),float(force[:,:,i,2].max())]))
        records.append(dict(label=label,per_body=per_body,input_hashes={name:sha(path/name) for name in ('metadata.json','trajectories.npz','sensor_diagnostic.npz')}))
result=dict(records=records,thresholds_changed=False,original_metrics_unchanged=True,
    sensor_is_ground_truth=False,
    interpretation='Solver sensors may include internal articulation reactions. Clear geometric separation diagnoses false ground-support readings; a stable sensor still requires support validation.',
    source_sha256=sha(Path(__file__)))
(OUT/'sensor_calibration_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
