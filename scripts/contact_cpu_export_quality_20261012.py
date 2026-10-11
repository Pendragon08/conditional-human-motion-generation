"""Component failure diagnosis on the stable export route, unchanged gates."""
import json,hashlib
from pathlib import Path
import numpy as np
from physics_guided_core import rotate_xyzw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_cpu_export_v11_20261012'
path=OUT/'raw/repeat_a'
meta=json.loads((path/'metadata.json').read_text())
rows=json.loads((path/'metrics.json').read_text())
t=json.loads((OUT/'frozen_thresholds.json').read_text())['thresholds']
components=[]
for row in rows:
    components.append(dict(case_id=row['case_id'],condition_pass=row['condition_pass'],
        penetration_pass=row['penetration_fraction']<=t['penetration_fraction'],
        surface_slip_proxy_pass=row['surface_slip_proxy_fraction'] is not None and row['surface_slip_proxy_fraction']<=t['sliding_fraction'],
        continuity_pass=row['jerk_p95_m_s3'] is not None and row['jerk_p95_m_s3']<=t['continuity_jerk_p95_m_s3'],
        slip_proxy_fraction=row['surface_slip_proxy_fraction'],penetration_fraction=row['penetration_fraction'],
        jerk_p95_m_s3=row['jerk_p95_m_s3'],tracking_pass=row['tracking_pass'],joint_pass=row['joint_pass']))
with np.load(path/'trajectories.npz',allow_pickle=False) as a:
    airborne=[]
    for name,g in meta['foot_geometry'].items():
        i=meta['names'].index(name)
        rotated=rotate_xyzw(a['quaternion'][:,:,i,None,:],np.asarray(g['points'])[None,None])
        height=(a['actual'][:,:,i,None,2]+rotated[...,2]-np.asarray(g['radii'])[None,None]).min(axis=-1)
        clear=(height>.05)&a['valid']
        f=a['contact_force'][:,:,i,:]
        airborne.append(dict(body=name,clear_geometric_separation_observations=int(clear.sum()),
            maximum_net_contact_norm_N=float(np.linalg.norm(f[clear],axis=-1).max()) if clear.any() else None,
            positive_vertical_above_frozen10N=int((f[:,:,2][clear]>t['contact_force_n']).sum()),
            note='Clear ground separation diagnostic; self/other-body contacts not ruled out; not a replacement ground truth label.'))
report=dict(component_pass_counts={k:sum(r[k] for r in components) for k in ['condition_pass','penetration_pass','surface_slip_proxy_pass','continuity_pass','tracking_pass','joint_pass']},
    inputs=len(rows),components=components,airborne_net_force_diagnostic=airborne,
    thresholds_changed=False,route_comparisons_must_be_rerun=True,
    full_measurement_gate_passed=False,innovation_confirmed=False,
    qualification='Stable exports and reset consistency pass; generated-motion quality still has a floor. Does not validate positive/negative humanoid quality discrimination or semantic retention.',
    input_hashes={x:hashlib.sha256((path/x).read_bytes()).hexdigest() for x in ['metadata.json','metrics.json','trajectories.npz']})
(OUT/'quality_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['component_pass_counts','inputs','airborne_net_force_diagnostic']},indent=2))
