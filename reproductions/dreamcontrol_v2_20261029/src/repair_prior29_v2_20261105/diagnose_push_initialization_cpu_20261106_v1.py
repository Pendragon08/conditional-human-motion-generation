"""Read-only CPU evidence for eight existing push pilots; no resampling."""
import json, hashlib, importlib.util, sys, datetime
from pathlib import Path
import numpy as np
S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
PLAN=S/'paper_pipeline_v10_launch/config.json'
OUT=S/'push_initialization_diagnostic_v1'
CID=20300007
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2,allow_nan=False)
def main():
    assert not OUT.exists(), 'Existing diagnostic preserved; inspect instead of restarting'
    assert sha(PLAN)=='e074e9b85998355a6588a0d1d711495014e587b724adaa5ee70a307b3986798a'
    plan=read(PLAN)
    for spec in list(plan['sources'].values())+[plan['urdf'],plan['fk']]:
        assert sha(spec['path'])==spec['sha256']
    assert read(S/'new_references_paper_v10/status.json')['status']=='failed_preserved'
    OUT.mkdir()
    sys.path.insert(0,str(S/'src/paper_protocol_v10'))
    import postopt_v2 as m
    spec=importlib.util.spec_from_file_location('push_independent_fk',plan['fk']['path'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    fk=mod.URDFFK(plan['urdf']['path']); links=sorted(fk.links)
    kp=[links.index(k) for k in mod.PAPER_22_LINKS]
    rows=[read(line_path['path'])['records'][0] for line_path in read(S/'new_references_paper_v10/progress.json')['cohorts'] if line_path['task']=='box_push_pull']
    selected=next(x for x in rows if x['candidate_id']==CID)
    sample=m.load_sample(selected['source_sample'],selected['source_sample_sha256'])
    legacy=S.parent/'stage3_supported_references_20261104/bulk_v1/library_records.jsonl'
    bank=[json.loads(x) for x in legacy.read_text().splitlines() if x.strip()]
    near=[]
    for r in bank:
        if r['task']!='box_push_pull': continue
        with np.load(r['path'],allow_pickle=False) as z: meta=json.loads(z['metadata_utf8'].tobytes())
        center=meta['task_object_position']
        near.append((float(np.linalg.norm(np.asarray(center)-sample['metadata']['task_object_position'])),r,meta))
    near.sort(key=lambda x:(x[0],x[1]['candidate_id']))
    comparisons=[]
    goal=sample['targets_world'][:,[20,21]]; mask=sample['wrist_mask']
    def stats(pts,source):
        wrist=pts[:,[20,21]];err=np.linalg.norm(wrist-goal,axis=-1)
        active=np.where(mask,err,-1);at=np.unravel_index(active.argmax(),active.shape)
        d=np.abs(wrist-[.45,0.,.4])-[.3,.3,.4]
        sdf=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(axis=-1),0)
        return dict(wrist_max_m=float(err[mask].max()),left_max_m=float(err[:,0][mask[:,0]].max()),right_max_m=float(err[:,1][mask[:,1]].max()),max_frame_hand=list(map(int,at)),wrist_at_max=wrist[at].tolist(),target_at_max=goal[at].tolist(),min_wrist_table_sdf_m=float(sdf[mask].min()),raw_rmse_m=m.rms_vector(pts-source['decoded_world']))
    for dist,r,meta in near[:12]:
        assert sha(r['path'])==r['sha256'] and sha(r['native_proof'])==r['native_proof_sha256']
        proof=read(r['native_proof']);assert proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values())
        assert meta['scene_sha256']==sample['metadata']['scene_sha256']
        assert meta['task_plan']['phases']==sample['metadata']['task_plan']['phases']
        with np.load(r['path'],allow_pickle=False) as z:
            names=z['joint_names'].tolist();q=z['q'].copy();root=np.c_[z['root_quat'],z['root_pos']]
            assert float(z['fps'])==20 and q.shape==(180,29)
        pts=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=links)[:,kp]
        comparisons.append(dict(seed=r,center_distance_m=dist,**stats(pts,sample)))
    chosen=min(comparisons,key=lambda x:(x['raw_rmse_m'],x['seed']['candidate_id']))
    states=[]
    rp=Path(selected['kinematic_report']);report=read(rp)
    assert sha(rp)==selected['kinematic_report_sha256']
    for path in [rp.parent/'stage_a_decoded_realization.npz']+[Path(x['path']) for x in report['stages']]:
        with np.load(path,allow_pickle=False) as z:q=z['q'].copy();root=z['root'].copy()
        pts=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=links)[:,kp]
        states.append(dict(path=str(path),sha256=sha(path),**stats(pts,sample)))
    seed=chosen['seed']
    with np.load(seed['path'],allow_pickle=False) as z:
        sq=z['q'].copy();sr=np.c_[z['root_quat'],z['root_pos']]
    with (OUT/'initial_state.npz').open('xb') as f:np.savez_compressed(f,q=sq,root=sr,joint_names=np.array(names))
    result=dict(scope='CPU only; fixed already generated candidate 20300007, nearest 12 old states checked as initialization only; no new samples or admissions',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_id=CID,new_samples=0,
        raw_path=selected['source_sample'],raw_sha256=selected['source_sample_sha256'],
        legacy_manifest_path=str(legacy),legacy_manifest_sha256=sha(legacy),
        seed_candidate_id=seed['candidate_id'],seed_source=seed['path'],seed_source_sha256=seed['sha256'],
        seed_native_proof=seed['native_proof'],seed_native_proof_sha256=seed['native_proof_sha256'],
        seed_state_sha256=sha(OUT/'initial_state.npz'),same_task_scene_and_phase=True,
        raw=stats(sample['decoded_world'],sample),stages=states,comparisons=comparisons,chosen=chosen)
    save(OUT/'cpu_report.json',result)
    print(json.dumps(dict(report=str(OUT/'cpu_report.json'),sha256=sha(OUT/'cpu_report.json'),raw=result['raw'],stages=states,chosen=chosen)))
if __name__=='__main__': main()

