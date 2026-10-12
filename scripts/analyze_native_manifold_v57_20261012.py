"""Contact material-point diagnosis on equality-checked V56 saved states.

No scores replaced: native PhysX observations are simulation-only diagnostic
evidence, not independent real contact/force ground truth. All 72 inputs in
each arm retained, including termination. No generator, policy or GPU call.
"""
from pathlib import Path
import json
import time
import numpy as np
from physics_guided_metric_v23_20261012 import rotate_xyzw
from verify_execution_lineage_v39_20261012 import sha,read,guard_batch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_manifold_v56_20261012'
OUT=ROOT/'results/native_manifold_analysis_v57_20261012'

def quantiles(values):
    return np.quantile(values,[.05,.5,.95]).tolist() if values else None

def describe(values,threshold):
    return dict(observations=len(values),quantiles_m_s=quantiles(values),
        above_frozen_speed=sum(x>threshold for x in values),
        fraction_above_frozen_speed=float(np.mean(np.asarray(values)>threshold)) if values else None)

def main():
    assert not OUT.exists(),'Preserve previous analysis; no automatic retry'
    assert np.__version__=='1.23.5'
    summary=read(BASE/'summary.json');assert summary['status']=='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS'
    t=read(BASE/'frozen_thresholds.json')['thresholds'];versions=read(BASE/'versions.json')
    assert sha(BASE/'frozen_thresholds.json')==versions['thresholds_sha256']
    inputs={str((BASE/'summary.json').relative_to(ROOT)):sha(BASE/'summary.json')};reports={}
    for arm in ('feedback','random'):
        path=BASE/'raw'/('val_%s_replay_s11'%arm)
        guard_batch(path,read(BASE/'requests'/(path.name+'.json')))
        eq=read(BASE/(arm+'_equality.json'))
        assert eq['frozen_metrics_exactly_equal'] and all(eq['exact_arrays'].values())
        meta=read(path/'metadata.json');native=read(path/'native_contacts.json');rows=read(path/'metrics.json')
        assert sha(path/'native_contacts.json')==meta['native_contacts_sha256'] and not native['errors']
        assert len(rows)==72 and len({r['source_id'] for r in rows})==24
        with np.load(path/'trajectories.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        names=meta['names'];feet={names.index(name):name for name in meta['foot_geometry']}
        point_groups={};heights=[];all_speed=[];non_ground=0;invalid_step=0
        for row in native['records']:
            if row['body1']==-1:body=row['body0'];side='0'
            elif row['body0']==-1:body=row['body1'];side='1'
            else:non_ground+=1;continue
            if body not in feet or row['force_magnitude_N']<=0:continue
            step=row['step'];n=row['environment']
            if step<meta['context_steps'] or not arrays['valid'][step,n]:invalid_step+=1;continue
            q=arrays['quaternion'][step,n,body];p=arrays['actual'][step,n,body]
            local=np.asarray(row['local_pos'+side])+np.asarray(row['offset'+side])
            lever=rotate_xyzw(q,local);com=rotate_xyzw(q,np.asarray(meta['native_body_properties'][n][body]['com']))
            world=p+lever
            v=arrays['linvel'][step,n,body]+np.cross(arrays['angvel'][step,n,body],lever-com)
            speed=float(np.linalg.norm(v[:2]));weight=float(row['force_magnitude_N'])
            heights.append(float(world[2]));all_speed.append(speed)
            point_groups.setdefault((n,step,body),[]).append((speed,weight))
        assert point_groups, 'No comparable native ground observations'
        cases=[];global_comparison=[];native_comparison=[]
        for n,row in enumerate(rows):
            length=int(arrays['valid'][:,n].sum());start=min(meta['context_steps'],length-1)
            idx=np.flatnonzero(arrays['valid'][:,n])[start:]
            proxy_values=[];matched=[];native_bodyframes=[];proxy_only=0
            for body,name in feet.items():
                g=meta['foot_geometry'][name];q=arrays['quaternion'][idx,n,body]
                p=arrays['actual'][idx,n,body];points=np.asarray(g['points']);radii=np.asarray(g['radii'])
                world=p[:,None]+rotate_xyzw(q[:,None],points[None]);world[:,:,2]-=radii[None]
                low=world[np.arange(len(idx)),np.argmin(world[:,:,2],axis=1)]
                com=rotate_xyzw(q,np.asarray(meta['native_body_properties'][n][body]['com']))
                vel=arrays['linvel'][idx,n,body]+np.cross(arrays['angvel'][idx,n,body],low-p-com)
                speed=np.linalg.norm(vel[:,:2],axis=1)
                supported=(arrays['contact_force'][idx,n,body,2]>t['contact_force_n'])&(low[:,2]<t['surface_near_ground_m'])
                for j,step in enumerate(idx):
                    if supported[j]:proxy_values.append(float(speed[j]))
                    contacts=point_groups.get((n,int(step),body),[])
                    if not contacts:
                        if supported[j]:proxy_only+=1
                        continue
                    speeds=np.array([x[0] for x in contacts]);weights=np.array([x[1] for x in contacts])
                    entry=dict(proxy=float(speed[j]),native_min=float(speeds.min()),native_max=float(speeds.max()),
                        native_weighted_mean=float(np.average(speeds,weights=weights)),supported=bool(supported[j]))
                    native_bodyframes.append(entry);native_comparison.append(entry)
                    if supported[j]:matched.append(entry);global_comparison.append(entry)
            fraction=float(np.mean(np.asarray(proxy_values)>t['sliding_speed_m_s'])) if proxy_values else None
            assert fraction==row['surface_slip_proxy_fraction'],'Original proxy fraction must reproduce exactly'
            s=t['sliding_speed_m_s']
            cases.append(dict(case_id=row['case_id'],source_id=row['source_id'],family=row['family'],offset_m=row['offset_m'],
                completed=row['completed'],original_joint_pass=row['joint_pass'],original_surface_slip_fraction=fraction,
                proxy_support_bodyframes=len(proxy_values),proxy_support_without_positive_native_ground=proxy_only,
                matched_support_bodyframes=len(matched),positive_native_ground_bodyframes=len(native_bodyframes),
                matched_proxy_above_speed=sum(x['proxy']>s for x in matched),
                matched_native_all_points_above_speed=sum(x['native_min']>s for x in matched),
                matched_native_any_point_above_speed=sum(x['native_max']>s for x in matched),
                proxy_above_but_all_native_points_at_or_below_speed=sum(x['proxy']>s and x['native_max']<=s for x in matched),
                proxy_at_or_below_but_all_native_points_above_speed=sum(x['proxy']<=s and x['native_min']>s for x in matched),
                native_point_diagnostic_means={key:float(np.mean([x[key] for x in native_bodyframes])) if native_bodyframes else None for key in ('native_min','native_max','native_weighted_mean')}))
        s=t['sliding_speed_m_s']
        reports[arm]=dict(cases=cases,total_cases=72,completed=sum(r['completed'] for r in rows),
            native_point_observations=len(all_speed),native_ground_height_quantiles_m=quantiles(heights),
            native_point_speed=describe(all_speed,s),proxy_matched_speed=describe([x['proxy'] for x in global_comparison],s),
            native_matched_min_speed=describe([x['native_min'] for x in global_comparison],s),
            native_matched_max_speed=describe([x['native_max'] for x in global_comparison],s),
            native_matched_weighted_speed=describe([x['native_weighted_mean'] for x in global_comparison],s),
            matched_support_bodyframes=len(global_comparison),positive_native_ground_bodyframes=len(native_comparison),
            proxy_above_but_all_native_points_at_or_below_speed=sum(x['proxy']>s and x['native_max']<=s for x in global_comparison),
            ignored_non_ground_contacts=non_ground,ignored_invalid_or_context_foot_ground_points=invalid_step,
            original_metrics_exactly_reproduced=True)
        for name in ('metadata.json','native_contacts.json','trajectories.npz','metrics.json','lineage.json'):
            inputs[str((path/name).relative_to(ROOT))]=sha(path/name)
        inputs[str((BASE/(arm+'_equality.json')).relative_to(ROOT))]=sha(BASE/(arm+'_equality.json'))
    result=dict(status='COMPLETE_NATIVE_CONTACT_MATERIAL_POINT_DIAGNOSIS',arms=reports,input_hashes=inputs,
        source_sha256=sha(Path(__file__)),generation_calls=0,simulation_calls=0,metrics_replaced=False,thresholds_changed=False,
        innovation_confirmed=False,overleaf_results_sync=False,evaluation_type='simulation_only; diagnostic contact-material-point proxy',
        scope='One validation seed, 24 sources per arm. Point counts are dependent within source and frame, not independent trials. Native material-point speeds are diagnostics, not replacement success scores. No real contact ground truth, force-control validation, controller-independent feasibility claim or candidate-efficacy claim.',time=time.time())
    OUT.mkdir();p=OUT/('diagnostic_%d.json'%time.time_ns());p.write_text(json.dumps(result,indent=2))
    print('NATIVE_CONTACT_MATERIAL_POINT_DIAGNOSIS_COMPLETE',p,flush=True)
    print(json.dumps({arm:{k:v for k,v in report.items() if k!='cases'} for arm,report in reports.items()},indent=2),flush=True)

if __name__=='__main__':main()
