"""Offline amended COM measurement on V17, without changing frozen thresholds."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_guided_metric_v21_20261012 import metrics, surface_metrics, rotate_xyzw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_com_metric_v21_20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Keep previous attempts'
    calibration=ROOT/'results/velocity_reference_fixture_v19_20261012/first_attempt'
    summary=json.loads((calibration/'summary.json').read_text())
    assert summary['status']=='PASS' and summary['exported_linear_velocity_reference']=='CENTER_OF_MASS'
    assert sha(calibration/'fixture_arrays.npz')==summary['arrays_sha256']
    with np.load(calibration/'fixture_arrays.npz',allow_pickle=False) as data:
        state=data['state'];com=data['com_local']
    point=np.array([.1,0,-.1]); lever=rotate_xyzw(state[:,:,3:7],point)
    world=state[:,:,:3]+lever
    numerical=np.gradient(world.astype(float),summary['dt'],axis=0)
    corrected=state[:,:,7:10]+np.cross(state[:,:,10:13],lever-rotate_xyzw(state[:,:,3:7],com[None]))
    legacy=state[:,:,7:10]+np.cross(state[:,:,10:13],lever)
    error=float(np.max(np.linalg.norm(corrected[2:-2]-numerical[2:-2],axis=-1)))
    legacy_error=float(np.max(np.linalg.norm(legacy[2:-2,1]-numerical[2:-2,1],axis=-1)))
    assert error<.003 and legacy_error>.05
    path=ROOT/'results/contact_point_human_v17_20261012/raw/repeat_a'
    meta=json.loads((path/'metadata.json').read_text())
    old=json.loads((path/'metrics.json').read_text())
    threshold_path=ROOT/'results/contact_point_human_v17_20261012/frozen_thresholds.json'
    thresholds=json.loads(threshold_path.read_text())['thresholds']
    with np.load(path/'trajectories.npz',allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    new=[]
    for i in range(len(meta['cases'])):
        current=json.loads(json.dumps(meta))
        for name,geometry in current['foot_geometry'].items():
            geometry['com_local']=meta['native_body_properties'][i][meta['names'].index(name)]['com']
        new.append(metrics(arrays,i,current,thresholds))
    geometry={'box':dict(points=[[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.05,.05)],radii=[0]*8,com_local=[0,0,0])}
    box_path=ROOT/'results/contact_force_fixture_v3_20261012/repeat_a/fixture_arrays.npz'
    with np.load(box_path,allow_pickle=False) as data:
        box=data['state'];force=data['net_contact_force']
    box_rows=[]
    for index,interval in [(0,slice(60,120)),(1,slice(60,120)),(2,slice(0,20))]:
        b=box[interval,index,None]
        box_rows.append(surface_metrics(b[...,:3],b[...,3:7],b[...,7:10],b[...,10:13],force[interval,index,None],['box'],geometry,thresholds))
    assert box_rows[0]['surface_slip_proxy_fraction']==0 and box_rows[1]['surface_slip_proxy_fraction']==1
    assert box_rows[2]['support_observations']==0
    OUT.mkdir()
    result=dict(status='OFFLINE_MEASUREMENT_AMENDMENT',material_point_corrected_error_max_m_s=error,
        material_point_legacy_error_max_m_s=legacy_error,box_control_rows=box_rows,
        cases=len(new),old_slip_pass=sum(r['surface_slip_proxy_fraction']<=thresholds['sliding_fraction'] for r in old),
        corrected_slip_pass=sum(r['surface_slip_proxy_fraction']<=thresholds['sliding_fraction'] for r in new),
        corrected_physical_pass=sum(r['physical_pass'] for r in new),corrected_joint_pass=sum(r['joint_pass'] for r in new),
        old=[dict(case_id=r['case_id'],surface_slip_proxy_fraction=r['surface_slip_proxy_fraction']) for r in old],corrected=new,
        thresholds_changed=False,old_results_retained=True,innovation_confirmed=False,
        full_humanoid_measurement_gate_pass=False,scope='Corrected velocity formula; articulated support calibration and full validation still required',
        files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'scripts/physics_guided_metric_v21_20261012.py',
            calibration/'summary.json',calibration/'fixture_arrays.npz',path/'metadata.json',path/'metrics.json',path/'trajectories.npz',threshold_path,box_path]})
    (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2))
    print('COM_CORRECTED_OFFLINE_MEASUREMENT_COMPLETE',len(new),result['corrected_slip_pass'],result['corrected_joint_pass'],flush=True)


if __name__=='__main__':
    main()
