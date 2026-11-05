"""Add two reviewed text candidates to an isolated, unaccepted preview.

The new clips share the existing SSM source family and are not independent
subjects. One source caption and one explicitly manual caption have bounded
kinematic support; this never grants source, physics or training acceptance.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from training_stats import training_stats
from provenance_groups import audit_provenance
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_punch_manual_preview_20261101';DATA=ROOT/'ssm_new6_20hz_resume_20261101';HERE=Path(__file__).resolve().parent
ORIG=ROOT/'ssm_007129_caption_review_20261101';MANUAL=ROOT/'ssm_004028_caption_review_20261101'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,data):
    with p.open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def copy(src,dst,want):
    if sha(src)!=want:raise ValueError('input changed')
    if dst.exists():
        if sha(dst)!=want:raise ValueError('duplicate path has different content')
        return
    with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    if sha(dst)!=want:raise ValueError('copy changed')
def main():
    torch.set_num_threads(1)
    records=json.loads((BASE/'manifest.json').read_text())['records'];states=json.loads((BASE/'state_manifest.json').read_text())['records'];cfg=json.loads((BASE/'config.json').read_text())
    dr=json.loads((DATA/'report.json').read_text());da=json.loads((DATA/'audit.json').read_text())
    if (sha(DATA/'report.json')!=da['source_report_sha256'] or sha(DATA/'records.jsonl')!=dr['records_sha256'] or
        da['technical_passed']!=5 or any(s in da['failed'] for s in ('007129','004028'))):
        raise ValueError('new data audit mismatch')
    data={r['id']:r for r in map(json.loads,(DATA/'records.jsonl').read_text().splitlines())}
    alignment=[json.loads(s) for s in Path(cfg['ssm_alignment_path']).read_text().splitlines()]
    source_bundle=json.loads(Path(cfg['ssm_source_evidence_manifest']).read_text())
    source_captions=json.loads(Path(cfg['caption_evidence_manifest']).read_text())
    manual=json.loads(Path(cfg['manual_caption_binding']['evidence_manifest']).read_text())
    for row in records:copy(BASE/row['feature_path'],HERE/row['feature_path'],row['feature_sha256'])
    for row in states:copy(BASE/row['state_path'],HERE/row['state_path'],row['state_sha256'])
    for sid,folder,caption_type in [('007129',ORIG,'source_text'),('004028',MANUAL,'manual_visual')]:
        match=[x for x in alignment if x.get('motion_id')==sid and x.get('mirrored') is False]
        if len(match)!=1:raise ValueError('alignment missing or duplicate')
        match=match[0];rec=data[sid];src=DATA/rec['artifact'];copy(src,HERE/(sid+'.npz'),rec['artifact_sha256'])
        if caption_type=='source_text':
            review=folder/(sid+'_review.json');check=folder/(sid+'_audit.json');review_data=json.loads(review.read_text());check_data=json.loads(check.read_text())
            if review_data['candidate_sha256']!=rec['artifact_sha256'] or check_data['receipt_sha256']!=sha(review):raise ValueError('source caption receipt mismatch')
            caption=review_data['caption'];source_note='HumanML3D texts/007129.txt; original line 3'
            source_captions['receipts'].append({'sample_id':sid,'receipt_path':str(review.relative_to(ROOT)),'receipt_sha256':sha(review),
                'independent_audit_path':str(check.relative_to(ROOT)),'independent_audit_sha256':sha(check)})
        else:
            review=folder/(sid+'_annotation.json');check=folder/(sid+'_audit.json');review_data=json.loads(review.read_text());check_data=json.loads(check.read_text())
            if review_data['feature_sha256']!=rec['artifact_sha256'] or check_data['annotation_sha256']!=sha(review):raise ValueError('manual caption receipt mismatch')
            caption=review_data['caption'];source_note='Codex manual annotation from curves; not source text'
            manual['receipts'].append({'sample_id':sid,'annotation_path':str(review.relative_to(ROOT)),'annotation_sha256':sha(review),
                'audit_path':str(check.relative_to(ROOT)),'audit_sha256':sha(check),'declared_source_fps':float(rec['source_fps_declared'])})
        records.append({'sample_id':sid,'source_id':match['source_path'],'group_id':'conservative-family:SSM_synced',
            'provenance_keys':{'source_path':[match['source_path']],'source_directory':[rec['source_group']],'source_family':['SSM_synced']},
            'split':'train','caption':caption,'caption_source':source_note,'caption_origin':caption_type,
            'caption_accepted':True,'caption_review_sha256':sha(review),'feature_path':sid+'.npz',
            'feature_sha256':rec['artifact_sha256'],'fps':20,'training_ready':False,
            'auxiliary_split_note':'same recording directory as prior SSM samples; no independent-subject claim'})
        states.append({'sample_id':sid,'feature_sha256':rec['artifact_sha256'],'state_path':sid+'.npz',
            'state_sha256':rec['artifact_sha256'],'foot_velocity_squared_threshold':.002})
        source_bundle['receipts'].append({'sample_id':sid,'receipt_sha256':sha(Path(match['source_receipt']))})
    if not audit_provenance(records)['source_groups_disjoint']:raise ValueError('provenance crosses split')
    train=[];spatial=[]
    for row in records:
        if row['split']!='train':continue
        with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
        train.append(('train',x))
        with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(float))
    stats=training_stats(train,min_std=.001);sp=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],spatial_mean=sp.mean(0),spatial_std=np.maximum(sp.std(0),.001))
    write(HERE/'manifest.json',{'records':records,'training_ready':False})
    write(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    write(HERE/'caption_evidence.json',source_captions);write(HERE/'manual_caption_evidence.json',manual);write(HERE/'source_evidence_manifest.json',source_bundle)
    cfg.update({'dataset_root':str(HERE),'manifest':str(HERE/'manifest.json'),'stats':str(HERE/'stats.npz'),
        'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json'),'run_dir':str(HERE),
        'caption_evidence_manifest':str(HERE/'caption_evidence.json'),'ssm_source_evidence_manifest':str(HERE/'source_evidence_manifest.json'),
        'purpose':'unaccepted eight-candidate technical preview',
        'manual_caption_binding':{'evidence_root':str(ROOT),'evidence_manifest':str(HERE/'manual_caption_evidence.json'),
                                  'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json')}})
    cfg['caption_archive_binding'].update({'evidence_manifest':str(HERE/'caption_evidence.json'),'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json')})
    cfg['vertical_support_diagnostic'].update({'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json')})
    write(HERE/'config.json',cfg)
    result={'base_manifest_sha256':sha(BASE/'manifest.json'),'new_data_report_sha256':sha(DATA/'report.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
        'stats_sha256':sha(HERE/'stats.npz'),'train_feature_frames':int(stats['frame_count']),'samples':len(records),
        'new_source_caption':'007129','new_manual_caption':'004028','training_ready':False,
        'limitations':['Both new samples share SSM family and recording directory; not independent subjects.',
            'Two reviews use kinematic curves, not full human/robot meshes or independent reviewers.',
            'Source, clock, physical quality and sufficient dataset/validation diversity remain unaccepted.']}
    write(HERE/'build_report.json',result);print(json.dumps(result))
if __name__=='__main__':main()
