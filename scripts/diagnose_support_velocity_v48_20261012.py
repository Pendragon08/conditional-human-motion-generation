"""Exploratory decomposition of frozen surface-slip proxy; not contact truth."""
import json
from pathlib import Path
import time
import numpy as np
from physics_guided_metric_v21_20261012 import rotate_xyzw
from verify_execution_lineage_v39_20261012 import sha,read,guard_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/support_velocity_diagnostic_v48_20261012'

def main():
    selected=(('planned','reference_velocity_v43_20261012','val_replay_s11'),
              ('native','native_reference_velocity_v46_20261012','val_native_replay_s11'),
              ('original','original_reference_velocity_v45_20261012','val_original_replay_s11'))
    reports={};inputs={}
    for arm,directory,label in selected:
        base=ROOT/'results'/directory;path=base/'raw'/label
        expected=read(base/'requests'/(label+'.json'));guard_batch(path,expected)
        meta=read(path/'metadata.json');rows=read(path/'metrics.json')
        t=read(base/'frozen_thresholds.json')['thresholds']
        for file in ('metadata.json','metrics.json','trajectories.npz','lineage.json'):
            inputs[str((path/file).relative_to(ROOT))]=sha(path/file)
        records=[];same_metric=True
        with np.load(path/'trajectories.npz',allow_pickle=False) as a:
            for n,row in enumerate(rows):
                length=int(a['valid'][:,n].sum());start=min(meta['context_steps'],length-1)
                idx=np.flatnonzero(a['valid'][:,n])[start:]
                supported_speeds=[]
                for name,g in meta['foot_geometry'].items():
                    k=meta['names'].index(name);q=a['quaternion'][idx,n,k]
                    pos=a['actual'][idx,n,k];lv=a['linvel'][idx,n,k];av=a['angvel'][idx,n,k]
                    points=np.asarray(g['points']);radii=np.asarray(g['radii'])
                    world=pos[:,None]+rotate_xyzw(q[:,None],points[None]);world[:,:,2]-=radii[None]
                    low=world[np.arange(len(idx)),np.argmin(world[:,:,2],axis=1)]
                    rotational=np.cross(av,low-pos-rotate_xyzw(q,np.asarray(g['com_local'])))
                    point_velocity=lv+rotational
                    speed=np.linalg.norm(point_velocity[:,:2],axis=1)
                    support=(a['contact_force'][idx,n,k,2]>t['contact_force_n'])&(low[:,2]<t['surface_near_ground_m'])
                    supported_speeds.extend(speed[support].tolist())
                    for j in np.flatnonzero(support):
                        records.append(dict(foot=name,family=row['family'],offset=row['offset_m'],
                            angular_speed=float(np.linalg.norm(av[j])),speed=float(speed[j]),
                            translation_xy=float(np.linalg.norm(lv[j,:2])),
                            rotation_xy=float(np.linalg.norm(rotational[j,:2])),
                            near_generation_boundary=abs(int(idx[j])-60)<=2))
                fraction=float(np.mean(np.asarray(supported_speeds)>t['sliding_speed_m_s'])) if supported_speeds else None
                assert (fraction is None and row['surface_slip_proxy_fraction'] is None) or abs(fraction-row['surface_slip_proxy_fraction'])<1e-10
        bins={}
        for lo,hi in ((0.,.25),(.25,1.),(1.,3.),(3.,float('inf'))):
            r=[x for x in records if lo<=x['angular_speed']<hi]
            bins[str(lo)+'-'+str(hi)]=dict(support_observations=len(r),
                above_frozen_slip_speed=sum(x['speed']>t['sliding_speed_m_s'] for x in r),
                mean_point_xy_speed=float(np.mean([x['speed'] for x in r])) if r else None,
                mean_com_xy_speed=float(np.mean([x['translation_xy'] for x in r])) if r else None,
                mean_rotational_xy_contribution=float(np.mean([x['rotation_xy'] for x in r])) if r else None)
        reports[arm]=dict(cases=72,support_observations=len(records),angular_bins_rad_s=bins,
            proxy_violations_total=sum(x['speed']>t['sliding_speed_m_s'] for x in records),
            proxy_violations_near_frame60=sum(x['speed']>t['sliding_speed_m_s'] and x['near_generation_boundary'] for x in records),
            frozen_slip_fractions_exactly_reproduced=True)
    result=dict(status='COMPLETE_EXPLORATORY_SUPPORT_VELOCITY_DECOMPOSITION',arms=reports,
        source_sha256=sha(Path(__file__)),metric_source_sha256=sha(ROOT/'scripts/physics_guided_metric_v21_20261012.py'),
        input_hashes=inputs,generation_calls=0,simulation_calls=0,thresholds_changed=False,
        innovation_confirmed=False,overleaf_results_sync=False,
        scope='Descriptive angular-speed bins were introduced after observing failures; not new acceptance rules. COM translation and angular contributions are not independent causal attribution. Closest collision-surface material point is not ground-truth contact point; no force control or actual sliding claim.')
    OUT.mkdir(exist_ok=True);path=OUT/ ('diagnostic_%d.json'%time.time_ns())
    path.write_text(json.dumps(result,indent=2));print(json.dumps(reports,indent=2));print('SUPPORT_DIAGNOSTIC',path)

if __name__=='__main__':
    main()
