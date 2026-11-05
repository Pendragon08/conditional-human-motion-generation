"""Build an explicitly provisional mixed-source 29-DoF, 20 Hz loader preview.

Uses already existing robot trajectories; no retargeting or physical filtering.
Candidate captions have not been semantically accepted and this is not training.
"""
import hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
T = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0] = [str(R), str(R/'robot_loader_20261030/src'), str(T)]
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from robot_motion_dataset import RobotMotionDataset, collate_robot_motion
from grouped_split import audit_assignments
from prepare_omni29_training_schema_20261103 import Moments

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    if shutil.disk_usage(R).free < 8*1024**3+356*1024**2: raise RuntimeError('disk reserve')
    out=R/'grab_omni_mixed29_full_preview_20261103'
    if out.exists(): raise FileExistsError(out)
    omni=R/'omni29_text_training_preview_20261103'
    evidence=json.loads((omni/'report.json').read_text())
    if sha(omni/'manifest.json') != evidence['manifest_sha256']: raise ValueError('Omni manifest changed')
    rows=json.loads((omni/'manifest.json').read_text())['records']
    if len(rows)!=1826 or any(r['caption_accepted'] or r['training_ready'] for r in rows): raise ValueError('unexpected Omni inputs')
    for r in rows: r['dataset_source']='OmniRetarget/OMOMO'
    seen=set(); dex_count=0
    for batch in ('batch10','batch100','batch1224'):
        d=R/('dexweave_stream_robot_'+batch+'_20261103')
        audit=json.loads((d/'audit.json').read_text())
        raw=(d/'records.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=audit['records_sha256'] or audit['passed']!=audit['samples']: raise ValueError('Dex audit binding')
        details={x['id']:x for x in audit['details']}
        for line in raw.splitlines():
            a=json.loads(line); sid=a['id']
            if sid in seen or a['status']!='technical_preview' or a['caption_accepted'] or a['training_ready']: raise ValueError('unexpected Dex row')
            if details[sid]['status']!='passed_technical_reread' or details[sid]['artifact_sha256']!=a['artifact_sha256']: raise ValueError('Dex audit row')
            seen.add(sid);subject=sid.split('/')[0]
            split='train' if subject in {'s1','s2','s3','s4','s5','s6','s7'} else 'val' if subject=='s8' else 'test' if subject in {'s9','s10'} else None
            if split is None or not a['captions']: raise ValueError('subject or captions')
            group='GRAB_nominal_subject:'+subject; source='GRAB_sequence:'+sid
            rows.append({'sample_id':'dex29_'+sid.replace('/','_'),'dataset_source':'DexWeave/GRAB',
                'group_id':group,'source_id':source,'split':split,'provenance_keys':{'nominal_subject':[group],
                'source_sequence':[source],'source_member_sha256':[a['source_member_sha256']]},
                'caption':a['captions'][0],'candidate_captions':a['captions'],
                'caption_source':'MotionHub GRAB hierarchical captions: '+sid,'caption_origin':'source_text',
                'caption_accepted':False,'training_ready':False,'annotation_sha256':a['text_sha256'],
                'feature_path':str((d/a['artifact']).relative_to(R)),'feature_sha256':a['artifact_sha256'],
                'state_path':str((d/a['artifact']).relative_to(R)),'state_sha256':a['artifact_sha256'],
                'fps':20.,'source_sha256':a['source_member_sha256'],'source_robot_sha256':a['input_sha256'],
                'encoding_variant':'sign_continuous_v1','cropped_for_max_10s':a['cropped_10_seconds'],
                'caption_risk_codes':a['risk_codes']})
            dex_count+=1
    if dex_count!=1334: raise ValueError('unexpected audited Dex count')
    # This previously audited singleton was intentionally excluded from all batches.
    d=R/'dexweave_grab_pilot_20261103'; a=json.loads((d/'preview_report.json').read_text())
    candidates=R/'grab_dex_candidate_pairs_20261103.jsonl'
    if sha(candidates)!='ba68debdf5061aaf7f3b5fa3cf8ab64c4b1266f439bca187cbbce5b13f849104': raise ValueError('candidate manifest hash')
    matching=[json.loads(line) for line in candidates.read_text(encoding='utf8').splitlines() if json.loads(line)['id']==a['id']]
    if len(matching)!=1 or a['id'] in seen: raise ValueError('singleton identity')
    c=matching[0]
    if c['source_member_sha256']!=a['source_member_sha256'] or c['text_sha256']!=a['caption_sha256'] or sha(d/'preview.npz')!=a['output_sha256']:raise ValueError('singleton binding')
    group='GRAB_nominal_subject:s9';source='GRAB_sequence:'+a['id'];seen.add(a['id'])
    rows.append({'sample_id':'dex29_'+a['id'].replace('/','_'),'dataset_source':'DexWeave/GRAB',
      'group_id':group,'source_id':source,'split':'test','provenance_keys':{'nominal_subject':[group],
      'source_sequence':[source],'source_member_sha256':[a['source_member_sha256']]},
      'caption':c['captions'][0],'candidate_captions':c['captions'],'caption_source':'MotionHub GRAB hierarchical captions: '+a['id'],
      'caption_origin':'source_text','caption_accepted':False,'training_ready':False,'annotation_sha256':a['caption_sha256'],
      'feature_path':str((d/'preview.npz').relative_to(R)),'feature_sha256':a['output_sha256'],
      'state_path':str((d/'preview.npz').relative_to(R)),'state_sha256':a['output_sha256'],
      'fps':20.,'source_sha256':a['source_member_sha256'],'source_robot_sha256':a['motion_sha256'],
      'encoding_variant':'sign_continuous_v1','cropped_for_max_10s':False,'caption_risk_codes':c['risk_codes']})
    dex_count+=1
    if dex_count!=1335 or len(seen)!=1335: raise ValueError('incomplete Dex coverage')
    audit_assignments(rows)
    fm,sm=Moments(263),Moments(66);counts=Counter();frames=Counter();crop=Counter()
    for i,r in enumerate(rows):
        if i%64==0 and shutil.disk_usage(R).free < 8*1024**3+356*1024**2: raise RuntimeError('disk reserve')
        p=R/r['feature_path']
        if sha(p)!=r['feature_sha256'] or r['fps']!=20.: raise ValueError('artifact or rate')
        with np.load(p,allow_pickle=False) as f:
            x=np.asarray(f['features'],dtype=np.float32)
            key=next((k for k in ('state29_wxyz_xyz','qpos29_source_layout_20hz') if k in f),None)
            if key is None: raise ValueError('missing 29-DoF state')
            state=np.asarray(f[key])
        if x.ndim!=2 or x.shape[1]!=263 or not 2<=len(x)<=199 or state.ndim!=2 or state.shape[1]<36 or len(state)!=len(x)+1 or not np.isfinite(x).all() or not np.isfinite(state).all(): raise ValueError('invalid source arrays')
        tag=r['dataset_source']+':'+r['split'];counts[tag]+=1;frames[tag]+=len(x)
        if r['cropped_for_max_10s']:crop[tag]+=1
        if r['split']=='train':
            with torch.no_grad(): pts=recover_from_ric(torch.from_numpy(x),22).numpy()
            if pts.shape!=(len(x),22,3) or not np.isfinite(pts).all(): raise ValueError('spatial decode')
            fm.add(x);sm.add(pts.reshape(len(x),66))
    out.mkdir(exist_ok=False)
    with (out/'manifest.json').open('x',encoding='utf8') as f:json.dump({'records':rows,'training_ready':False},f,ensure_ascii=False,allow_nan=False)
    fmean,fstd=fm.result();smean,sstd=sm.result()
    with (out/'stats.npz').open('xb') as f:np.savez(f,feature_mean=fmean,feature_std=fstd,spatial_mean=smean,spatial_std=sstd)
    stats={'feature_mean':fmean,'feature_std':fstd,'spatial_mean':smean,'spatial_std':sstd};cases=[]
    for split in ('train','val','test'):
        ds=RobotMotionDataset(root=R,records=rows,split=split,**stats)
        for source in ('OmniRetarget/OMOMO','DexWeave/GRAB'):
            j=next(i for i,r in enumerate(ds.records) if r['dataset_source']==source)
            item=ds[j];motion,condition=collate_robot_motion([item])
            if not torch.isfinite(motion).all() or not torch.isfinite(condition['y']['hint']).all():raise ValueError('loader')
            cases.append({'split':split,'source':source,'sample_id':item['sample_id'],'feature_frames':len(item['motion'])})
    report={'manifest_sha256':sha(out/'manifest.json'),'stats_sha256':sha(out/'stats.npz'),
        'inputs':{'omni_manifest_sha256':evidence['manifest_sha256'],'dex_already_audited_samples':dex_count},
        'samples_by_source_split':dict(counts),'frames_by_source_split':dict(frames),'cropped_by_source_split':dict(crop),
        'train_frames':fm.n,'loader_cases':cases,'accepted_captions':0,'training_ready':False,'physical_action_filter':False,
        'limitations':['Captions, source licensing, model compatibility and time basis are not semantically accepted.',
                       'Grouping uses nominal source subjects; independence across datasets is unproven.',
                       '10-second truncation may invalidate full-sequence captions.',
                       '29 joint states remain in referenced NPZ files; 263D features do not retain all joint information.',
                       'Stats are provisional and must be recalculated once accepted data are chosen.']}
    with (out/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
