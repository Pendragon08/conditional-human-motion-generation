"""Offline native contact / velocity-reference diagnosis; not score replacement."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_guided_core import rotate_xyzw

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'results/contact_point_human_v17_20261012/raw/repeat_a'
OUT = ROOT/'results/contact_point_analysis_v18_20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(values):
    return np.quantile(values, [.05,.5,.95]).tolist() if len(values) else None


def main():
    assert not OUT.exists(), 'Preserve every analysis attempt'
    OUT.mkdir()
    meta=json.loads((INPUT/'metadata.json').read_text())
    native=json.loads((INPUT/'native_contacts.json').read_text())
    assert sha(INPUT/'native_contacts.json')==meta['native_contacts_sha256']
    assert not native['errors'], 'Native parser must be valid before analysis'
    with np.load(INPUT/'trajectories.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    names=meta['names']; foot=[names.index(name) for name in meta['foot_geometry']]
    com=np.asarray([[body['com'] for body in row] for row in meta['native_body_properties']])
    assert com.shape==(arrays['actual'].shape[1],24,3)
    rotated_com=rotate_xyzw(arrays['quaternion'],com[None])
    exported=arrays['linvel'].astype(float)
    origin_derivative=np.gradient(arrays['actual'].astype(float),meta['dt'],axis=0)
    com_derivative=np.gradient(arrays['actual'].astype(float)+rotated_com,meta['dt'],axis=0)
    per_case=[]; records=[]
    for row in native['records']:
        step=row['step']; i=row['environment']
        if step<meta['context_steps'] or not arrays['valid'][step,i]:
            continue
        if row['force_magnitude_N']<=0:
            continue
        if row['body1']==-1:
            body=row['body0']; side='0'
        elif row['body0']==-1:
            body=row['body1']; side='1'
        else:
            continue
        if body not in foot:
            continue
        local=np.asarray(row['local_pos'+side])+np.asarray(row['offset'+side])
        lever=rotate_xyzw(arrays['quaternion'][step,i,body],local)
        world=arrays['actual'][step,i,body]+lever
        velocity_origin=exported[step,i,body]+np.cross(arrays['angvel'][step,i,body],lever)
        velocity_com=exported[step,i,body]+np.cross(arrays['angvel'][step,i,body],lever-rotated_com[step,i,body])
        # Diagnostic hypotheses only: neither becomes the frozen score here.
        records.append(dict(case=i,step=step,body=body,world_height_m=float(world[2]),
            force_N=row['force_magnitude_N'],speed_origin_hypothesis_m_s=float(np.linalg.norm(velocity_origin[:2])),
            speed_com_hypothesis_m_s=float(np.linalg.norm(velocity_com[:2]))))
    threshold=json.loads((ROOT/'results/contact_point_human_v17_20261012/frozen_thresholds.json').read_text())['thresholds']
    metrics=json.loads((INPUT/'metrics.json').read_text())
    for i,case in enumerate(meta['cases']):
        rows=[row for row in records if row['case']==i]
        active=arrays['valid'][:,i].copy();active[:meta['context_steps']]=False
        indices=np.flatnonzero(active)
        # Central difference values adjacent to a reset/termination are excluded.
        indices=indices[(indices>0)&(indices<arrays['valid'].shape[0]-1)]
        indices=indices[arrays['valid'][indices-1,i]&arrays['valid'][indices+1,i]]
        origin_error=np.linalg.norm(origin_derivative[indices,i][:,foot]-exported[indices,i][:,foot],axis=-1).ravel()
        com_error=np.linalg.norm(com_derivative[indices,i][:,foot]-exported[indices,i][:,foot],axis=-1).ravel()
        item=dict(case_id=case['case_id'],native_positive_ground_points=len(rows),
            original_surface_slip_fraction=metrics[i]['surface_slip_proxy_fraction'],
            native_ground_height_quantiles_m=q([row['world_height_m'] for row in rows]),
            origin_velocity_finite_difference_error_quantiles_m_s=q(origin_error),
            com_velocity_finite_difference_error_quantiles_m_s=q(com_error))
        for hypothesis in ('origin','com'):
            speeds=np.asarray([row['speed_'+hypothesis+'_hypothesis_m_s'] for row in rows])
            item[hypothesis+'_contact_speed_quantiles_m_s']=q(speeds)
            item[hypothesis+'_diagnostic_point_fraction_above_5cm_s']=float((speeds>threshold['sliding_speed_m_s']).mean()) if len(speeds) else None
        per_case.append(item)
    result=dict(cases=per_case,positive_ground_records=len(records),native_total_records=len(native['records']),
        com_local_foot_m=com[:,foot].tolist(),native_parser_errors=0,metrics_replaced=False,thresholds_changed=False,
        innovation_confirmed=False,scope='Contact-point and velocity-reference hypotheses only; finite differences at30Hz are not independent calibration',
        files={str((INPUT/name).relative_to(ROOT)):sha(INPUT/name) for name in ('native_contacts.json','metadata.json','trajectories.npz','metrics.json')},
        source_sha256=sha(Path(__file__)))
    (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_HUMANOID_CONTACT_OFFLINE_DIAGNOSIS_COMPLETE',len(per_case),len(records),flush=True)


if __name__=='__main__':
    main()
