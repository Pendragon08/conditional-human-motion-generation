"""CPU preparation for a new validation-only causal probe. Preserve V51 data."""
from pathlib import Path
import json
import shutil
import numpy as np
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_metric_v23_20261012 import rotate_xyzw
from support_reference_repair_core_v62_20261012 import translation,numeric_checks

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
OUT=ROOT/'results/support_reference_probe_v62_20261012'


def prepare():
    assert __debug__ and np.__version__=='1.23.5'
    assert not OUT.exists(),'Prior attempt retained; no preparation rerun'
    numeric=numeric_checks()
    source=BASE/'raw/val_shared_s11_r0'
    binding=guard_batch(source,read(BASE/'requests/val_shared_s11_r0.json'))
    meta=read(source/'metadata.json');rows=read(source/'metrics.json')
    thresholds=read(BASE/'frozen_thresholds.json')['thresholds']
    with np.load(source/'trajectories.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    assert len(rows)==72 and len({r['source_id'] for r in rows})==24
    refs=np.empty((120,72,24,3),np.float32);motions={};records=[]
    for n,row in enumerate(rows):
        length=int(a['valid'][:,n].sum());assert length>0
        refs[:length,n]=a['actual'][:length,n];refs[length:,n]=a['actual'][length-1,n]
    motions['baseline']=refs
    for arm,endpoint in (('unconstrained',False),('endpoint_preserved',True)):
        repaired=refs.copy();arm_rows=[]
        for n,row in enumerate(rows):
            length=int(a['valid'][:,n].sum());start=meta['context_steps']
            velocity=np.zeros((120,4,2));support=np.zeros((120,4),bool)
            for j,(name,g) in enumerate(meta['foot_geometry'].items()):
                body=meta['names'].index(name);idx=np.arange(length)
                q=a['quaternion'][idx,n,body];p=a['actual'][idx,n,body]
                world=p[:,None]+rotate_xyzw(q[:,None],np.asarray(g['points'])[None])
                world[:,:,2]-=np.asarray(g['radii'])[None]
                low=world[np.arange(length),np.argmin(world[:,:,2],axis=1)]
                com=rotate_xyzw(q,np.asarray(meta['native_body_properties'][n][body]['com']))
                v=a['linvel'][idx,n,body]+np.cross(a['angvel'][idx,n,body],low-p-com)
                velocity[:length,j]=v[:,:2]
                support[:length,j]=(a['contact_force'][idx,n,body,2]>thresholds['contact_force_n'])&(low[:,2]<thresholds['surface_near_ground_m'])
            # Incomplete inputs remain in denominator; never invent contact
            # information for a padded tail or perform repairs from invalid data.
            if length==120:
                delta=translation(velocity,support,start,1/30,endpoint)
            else:delta=np.zeros((120,2))
            repaired[:,n,:,:2]+=delta[:,None].astype(np.float32)
            assert np.array_equal(repaired[:start+1,n],refs[:start+1,n])
            assert np.array_equal(repaired[:,n,:,2],refs[:,n,:,2])
            relative=(repaired[:,n]-repaired[:,n,0:1])-(refs[:,n]-refs[:,n,0:1])
            assert np.max(np.abs(relative))<2e-6
            if endpoint:assert np.array_equal(repaired[-1,n],refs[-1,n])
            arm_rows.append(dict(case_id=row['case_id'],source_id=row['source_id'],construction_completed=row['completed'],
                complete_input=length==120,max_translation_m=float(np.linalg.norm(delta,axis=1).max()),
                endpoint_translation_m=float(np.linalg.norm(delta[-1])),relative_pose_roundoff_m=float(np.abs(relative).max())))
        motions[arm]=repaired;records.append(dict(arm=arm,cases=arm_rows))
    OUT.mkdir();(OUT/'references').mkdir();(OUT/'requests').mkdir();(OUT/'raw').mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):shutil.copyfile(BASE/name,OUT/name)
    for arm,reference in motions.items():
        np.savez_compressed(OUT/'references'/(arm+'.npz'),reference=reference,original_goal=np.asarray(meta['original_goal'],np.float32))
    files=dict(read(BASE/'versions.json')['files'])
    for name in ('scripts/support_reference_repair_core_v62_20261012.py','scripts/prepare_support_reference_probe_v62_20261012.py',
                 'scripts/run_support_reference_probe_v62_20261012.py','docs/support_reference_probe_v62_20261012.md'):
        files[name]=sha(ROOT/name)
    save=lambda p,x:p.write_text(json.dumps(x,indent=2),encoding='utf8')
    save(OUT/'versions.json',dict(files=files,source_manifest_sha256=sha(OUT/'source_manifest.json'),thresholds_sha256=sha(OUT/'frozen_thresholds.json'),
        parent_versions_sha256=sha(BASE/'versions.json'),controller_weights_sha256=read(BASE/'versions.json')['controller_weights_sha256'],
        generator_weights_sha256=read(BASE/'versions.json')['generator_weights_sha256'],evaluation_type='simulation_only; synthetic_proxy tracking',
        scope='Exploratory reference-side intervention, one validation seed; no candidate efficacy or novel-method claim',
        confirmatory=False,innovation_confirmed=False,overleaf_results_sync=False))
    save(OUT/'preparation.json',dict(status='PREPARED',numeric_checks=numeric,source_binding=binding,source_metrics_sha256=sha(source/'metrics.json'),
        source_case_order=[r['case_id'] for r in rows],source_original_goal_sha256=meta['original_goal_sha256'],
        baseline_reference_sha256=sha(OUT/'references/baseline.npz'),reference_hashes={arm:sha(OUT/'references'/(arm+'.npz')) for arm in motions},
        repairs=records,original_targets_changed=False,source_files_changed=False,simulation_calls=0,generation_calls=0))
    save(OUT/'budget_ledger.json',[])
    print('SUPPORT_REFERENCE_V62_PREPARED',OUT,flush=True)


if __name__=='__main__':prepare()
