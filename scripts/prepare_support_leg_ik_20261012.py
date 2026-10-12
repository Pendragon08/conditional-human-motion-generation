"""Prepare a limited positional-IK diagnosis without changing source records."""
from pathlib import Path
import json
import shutil
import numpy as np
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_metric_v23_20261012 import rotate_xyzw
from support_reference_repair_core_v63_20261012 import translation
from support_leg_ik_core_20261012 import knee_solution,numeric_checks

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/support_reference_probe_v63_20261012'
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
OUT=ROOT/'results/support_leg_ik_20261012'


def prepare():
    if OUT.exists():
        raise RuntimeError('Do not overwrite an earlier attempt')
    checks=numeric_checks()
    source=BASE/'raw/val_shared_s11_r0'
    source_binding=guard_batch(source,read(BASE/'requests/val_shared_s11_r0.json'))
    baseline=PARENT/'raw/val_baseline_s11'
    baseline_binding=guard_batch(baseline,read(PARENT/'requests/val_baseline_s11.json'))
    metadata=read(source/'metadata.json');rows=read(source/'metrics.json')
    thresholds=read(PARENT/'frozen_thresholds.json')['thresholds']
    with np.load(source/'trajectories.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    with np.load(PARENT/'references/baseline.npz',allow_pickle=False) as z:
        refs=z['reference'].copy();goals=z['original_goal'].copy()
    if refs.shape!=(120,72,24,3) or len({r['source_id'] for r in rows})!=24:
        raise RuntimeError('Unexpected frozen input scope')
    for case in metadata['cases']:
        if case['goal_joint_body'] not in ('Pelvis','L_Wrist','R_Wrist'):
            raise RuntimeError('This positional ablation must leave task-body coordinates unchanged')
    cartesian=refs.copy();ik=refs.copy();records=[]
    for n,row in enumerate(rows):
        length=int(arrays['valid'][:,n].sum());start=metadata['context_steps']
        for side in ('L','R'):
            hip,knee,ankle,toe=[metadata['names'].index(side+'_'+part) for part in ('Hip','Knee','Ankle','Toe')]
            velocity=np.zeros((120,2,2));support=np.zeros((120,2),bool)
            for j,body in enumerate((ankle,toe)):
                name=metadata['names'][body];geom=metadata['foot_geometry'][name]
                idx=np.arange(length);q=arrays['quaternion'][idx,n,body];p=arrays['actual'][idx,n,body]
                points=p[:,None]+rotate_xyzw(q[:,None],np.asarray(geom['points'])[None])
                points[:,:,2]-=np.asarray(geom['radii'])[None]
                low=points[np.arange(length),np.argmin(points[:,:,2],axis=1)]
                com=rotate_xyzw(q,np.asarray(metadata['native_body_properties'][n][body]['com']))
                v=arrays['linvel'][idx,n,body]+np.cross(arrays['angvel'][idx,n,body],low-p-com)
                velocity[:length,j]=v[:,:2]
                support[:length,j]=(arrays['contact_force'][idx,n,body,2]>thresholds['contact_force_n'])&(low[:,2]<thresholds['surface_near_ground_m'])
            requested=translation(velocity,support,start,1/30,True) if length==120 else np.zeros((120,2))
            side_rows=[]
            for t in range(start+1,119):
                result,delta,alpha=knee_solution(refs[t,n,hip],refs[t,n,knee],refs[t,n,ankle],np.r_[requested[t],0.])
                for body in (ankle,toe):
                    cartesian[t,n,body]=refs[t,n,body]+delta.astype(np.float32)
                    ik[t,n,body]=cartesian[t,n,body]
                ik[t,n,knee]=result.astype(np.float32)
                # Float32 positional constraints, not whole-body dynamical feasibility.
                before=[np.linalg.norm(refs[t,n,knee]-refs[t,n,hip]),np.linalg.norm(refs[t,n,ankle]-refs[t,n,knee])]
                after=[np.linalg.norm(ik[t,n,knee]-ik[t,n,hip]),np.linalg.norm(ik[t,n,ankle]-ik[t,n,knee])]
                err=max(abs(float(a)-float(b)) for a,b in zip(before,after))
                if err>1e-5 or np.linalg.norm(ik[t,n,knee]-refs[t,n,knee])>.10001:
                    raise RuntimeError('Joint-coordinate bounds failed')
                side_rows.append(dict(frame=t,scale=alpha,accepted_shift_m=float(np.linalg.norm(delta)),bone_error_m=err))
            records.append(dict(case_id=row['case_id'],source_id=row['source_id'],side=side,frames=side_rows))
    altered=[metadata['names'].index(s+'_'+part) for s in ('L','R') for part in ('Knee','Ankle','Toe')]
    unaltered=[i for i in range(24) if i not in altered]
    for arm in (cartesian,ik):
        if not np.array_equal(arm[:start+1],refs[:start+1]) or not np.array_equal(arm[-1],refs[-1]) or not np.array_equal(arm[:,:,unaltered],refs[:,:,unaltered]):
            raise RuntimeError('Prefix, endpoint or task/body anchor changed')
    OUT.mkdir();(OUT/'references').mkdir();(OUT/'requests').mkdir();(OUT/'raw').mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(PARENT/name,OUT/name)
    shutil.copyfile(PARENT/'references/baseline.npz',OUT/'references/baseline.npz')
    for arm,reference in (('cartesian',cartesian),('two_link_ik',ik)):
        np.savez_compressed(OUT/'references'/(arm+'.npz'),reference=reference,original_goal=goals)
    files=dict(read(PARENT/'versions.json')['files'])
    for name in ('scripts/support_leg_ik_core_20261012.py','scripts/prepare_support_leg_ik_20261012.py','scripts/run_support_leg_ik_20261012.py','docs/support_leg_ik_20261012.md'):
        files[name]=sha(ROOT/name)
    version=dict(read(PARENT/'versions.json'),files=files,source_manifest_sha256=sha(OUT/'source_manifest.json'),thresholds_sha256=sha(OUT/'frozen_thresholds.json'),parent_versions_sha256=sha(PARENT/'versions.json'),scope='One-seed internal-validation positional-IK diagnosis, not candidate efficacy')
    save=lambda p,x:p.write_text(json.dumps(x,indent=2),encoding='utf8')
    save(OUT/'versions.json',version)
    save(OUT/'preparation.json',dict(status='PREPARED',checks=checks,records=records,source_binding=source_binding,baseline_binding=baseline_binding,
        source_original_goal_sha256=metadata['original_goal_sha256'],source_case_order=[r['case_id'] for r in rows],reference_hashes={a:sha(OUT/'references'/(a+'.npz')) for a in ('baseline','cartesian','two_link_ik')},
        original_targets_changed=False,simulation_calls=0,generation_calls=0,innovation_confirmed=False))
    save(OUT/'budget_ledger.json',[])
    print('SUPPORT_LEG_IK_PREPARED',flush=True)


if __name__=='__main__':prepare()
