"""Read-only velocity decomposition; not an alternative quality metric."""
import json
import time
from pathlib import Path
import numpy as np
from prepare_support_leg_ik_20261012 import ROOT,OUT,PARENT
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_metric_v23_20261012 import rotate_xyzw


def main():
    if read(OUT/'summary.json')['status']!='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS':
        raise RuntimeError('Complete fixed-arm records required')
    thresholds=read(OUT/'frozen_thresholds.json')['thresholds']
    bindings={};reports={}
    def bind(p):bindings[str(p.relative_to(ROOT))]=sha(p)
    bind(OUT/'summary.json');bind(OUT/'frozen_thresholds.json')
    with np.load(OUT/'references/baseline.npz',allow_pickle=False) as z:base_ref=z['reference']
    reference_axis_error={}
    for arm in ('baseline','cartesian','two_link_ik'):
        home=PARENT if arm=='baseline' else OUT
        folder=home/'raw'/('val_'+arm+'_s11');request=home/'requests'/(folder.name+'.json')
        guard_batch(folder,read(request));meta=read(folder/'metadata.json');stored=read(folder/'metrics.json')
        for name in ('metadata.json','metrics.json','trajectories.npz','receipt.json','lineage.json'):bind(folder/name)
        bind(request);bind(OUT/'references'/(arm+'.npz'))
        with np.load(folder/'trajectories.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        with np.load(OUT/'references'/(arm+'.npz'),allow_pickle=False) as z:reference=z['reference']
        errors=[]
        for side in ('L','R'):
            a,t=[meta['names'].index(side+'_'+part) for part in ('Ankle','Toe')]
            errors.append(float(np.max(np.abs((reference[:,:,t]-reference[:,:,a])-(base_ref[:,:,t]-base_ref[:,:,a])))))
        reference_axis_error[arm]=max(errors)
        rows=[]
        for n,case in enumerate(meta['cases']):
            valid=np.flatnonzero(arrays['valid'][:,n]);ids=valid[meta['context_steps']:]
            groups=[];total=bad=0
            for name,geom in meta['foot_geometry'].items():
                body=meta['names'].index(name);p=arrays['actual'][ids,n,body];q=arrays['quaternion'][ids,n,body]
                world=p[:,None]+rotate_xyzw(q[:,None],np.asarray(geom['points'])[None])
                world[:,:,2]-=np.asarray(geom['radii'])[None]
                low=world[np.arange(len(ids)),np.argmin(world[:,:,2],axis=1)]
                com=rotate_xyzw(q,np.asarray(meta['native_body_properties'][n][body]['com']))
                linear=arrays['linvel'][ids,n,body,:2]
                rotational=np.cross(arrays['angvel'][ids,n,body],low-p-com)[:,:2]
                material=linear+rotational
                support=(arrays['contact_force'][ids,n,body,2]>thresholds['contact_force_n'])&(low[:,2]<thresholds['surface_near_ground_m'])
                speed=np.linalg.norm(material,axis=1);count=int(support.sum());slip=int((support&(speed>thresholds['sliding_speed_m_s'])).sum())
                total+=count;bad+=slip
                moving=support&(speed>1e-12)
                aligned_linear=np.einsum('ij,ij->i',linear[moving],material[moving])/speed[moving]
                aligned_rot=np.einsum('ij,ij->i',rotational[moving],material[moving])/speed[moving]
                groups.append(dict(body=name,support_observations=count,slip_proxy_observations=slip,
                    mean_COM_linear_speed_m_s=float(np.linalg.norm(linear[support],axis=1).mean()) if count else None,
                    mean_rotational_term_speed_m_s=float(np.linalg.norm(rotational[support],axis=1).mean()) if count else None,
                    mean_material_speed_m_s=float(speed[support].mean()) if count else None,
                    mean_signed_linear_projection_m_s=float(aligned_linear.mean()) if moving.any() else None,
                    mean_signed_rotational_projection_m_s=float(aligned_rot.mean()) if moving.any() else None,
                    note='Signed projections can cancel; term magnitudes are not additive and not standalone slip metrics'))
            ratio=bad/total if total else None
            if (ratio is None)!=(stored[n]['surface_slip_proxy_fraction'] is None) or (ratio is not None and ratio!=stored[n]['surface_slip_proxy_fraction']):
                raise RuntimeError('Exact frozen proxy count mismatch')
            rows.append(dict(case_id=case['case_id'],source_id=case['source_id'],support_observations=total,
                slip_proxy_observations=bad,surface_slip_proxy_fraction=ratio,bodies=groups))
        groups=[g for r in rows for g in r['bodies'] if g['support_observations']]
        total=sum(g['support_observations'] for g in groups)
        mean=lambda key:sum(g[key]*g['support_observations'] for g in groups)/total if total else None
        reports[arm]=dict(inputs=len(rows),rows=rows,support_observations=total,
            pooled_slip_proxy_fraction=sum(r['slip_proxy_observations'] for r in rows)/total if total else None,
            support_weighted_mean_COM_linear_speed_m_s=mean('mean_COM_linear_speed_m_s'),
            support_weighted_mean_rotational_term_speed_m_s=mean('mean_rotational_term_speed_m_s'),
            support_weighted_mean_material_speed_m_s=mean('mean_material_speed_m_s'))
    target=OUT/('velocity_components_%d.json'%time.time_ns())
    result=dict(status='COMPLETE_READONLY_COMPONENT_DIAGNOSIS',reports=reports,reference_toe_ankle_vector_max_change_m=reference_axis_error,
        input_hashes=bindings,source_sha256=sha(Path(__file__)),simulation_calls=0,generation_calls=0,training_calls=0,
        innovation_confirmed=False,candidate_efficacy_test=False,thresholds_changed=False,
        limitations='One-seed proxy decomposition; no contact-point truth, causal isolation, alternative score or efficacy acceptance')
    with target.open('x') as f:json.dump(result,f,indent=2)
    print('LEG_IK_VELOCITY_COMPONENT_DIAGNOSIS_COMPLETE',target,flush=True)
    print(json.dumps({a:{k:v for k,v in r.items() if k!='rows'} for a,r in reports.items()},indent=2),flush=True)


if __name__=='__main__':main()
