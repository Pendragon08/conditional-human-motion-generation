"""Build a five-case auxiliary preflight dataset, never a training dataset."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from training_stats import training_stats
from provenance_groups import audit_provenance

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'four_semantic_subset_20261101';DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
GRAB=ROOT/'grab_s3_annotated_preview_20261101';ANNOT=ROOT/'grab_s3_manual_caption_20261101'
PRIOR=ROOT/'caption_archive_preflight_20261101/config.json';HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def copy(src,dst,expected):
    if sha(src)!=expected:raise ValueError('source hash mismatch')
    if dst.exists():
        if sha(dst)!=expected:raise ValueError('conflicting repeated artifact')
        return
    with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    if sha(dst)!=expected:raise ValueError('copy hash mismatch')

def main():
    torch.set_num_threads(1)
    records=json.loads((BASE/'manifest.json').read_text())['records']
    states=json.loads((BASE/'state_manifest.json').read_text())['records']
    original_cfg=json.loads(PRIOR.read_text());grab=json.loads((GRAB/'manifest.json').read_text())
    grab_audit=json.loads((GRAB/'audit.json').read_text());r=grab['records'][0]
    annotation=json.loads((ANNOT/'annotation.json').read_text());aa=json.loads((ANNOT/'audit.json').read_text())
    if (sha(GRAB/'manifest.json')!=grab_audit['manifest_sha256'] or sha(ANNOT/'annotation.json')!=aa['annotation_sha256'] or
        r['manual_caption_receipt_sha256']!=aa['annotation_sha256'] or grab_audit['state_audit']['state_technical_checks_passed'] is not True):
        raise ValueError('GRAB evidence mismatch')
    if annotation['caption']!=r['caption'] or r['split']!='unassigned':raise ValueError('unexpected GRAB caption or prior split')
    for row in records:copy(DATA/row['feature_path'],HERE/row['feature_path'],row['feature_sha256'])
    for row in states:copy(DATA/row['state_path'],HERE/row['state_path'],row['state_sha256'])
    sid=r['sample_id'];feature='GRAB_S3_features.npz';state='GRAB_S3_state.npz'
    copy(GRAB/r['feature_path'],HERE/feature,r['feature_sha256']);copy(GRAB/r['state_path'],HERE/state,r['state_sha256'])
    records.append({'sample_id':sid,'source_id':r['source_id'],'group_id':'conservative-family:GRAB',
        'provenance_keys':{'source_path':[r['source_id']],'source_directory':['GRAB:s3'],'source_family':['GRAB']},
        'split':'train','caption':r['caption'],'caption_source':r['caption_source'],'caption_origin':'manual_visual',
        'caption_accepted':True,'caption_review_sha256':sha(ANNOT/'annotation.json'),
        'feature_path':feature,'feature_sha256':r['feature_sha256'],'fps':20,'training_ready':False,
        'auxiliary_split_note':'diagnostic family grouping; not author split and not proof of independence'})
    states.append({'sample_id':sid,'feature_sha256':r['feature_sha256'],'state_path':state,
                   'state_sha256':r['state_sha256'],'foot_velocity_squared_threshold':r['foot_velocity_squared_threshold']})
    if not audit_provenance(records)['source_groups_disjoint']:raise ValueError('known provenance crosses split')
    train=[];spatial=[]
    for row in records:
        if row['split']!='train':continue
        with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
        train.append(('train',x))
        with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(float))
    stats=training_stats(train,min_std=.001);p=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],spatial_mean=p.mean(0),spatial_std=np.maximum(p.std(0),.001))
    save(HERE/'manifest.json',{'records':records,'training_ready':False});save(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    save(HERE/'manual_caption_evidence.json',{'receipts':[{'sample_id':sid,
        'annotation_path':'grab_s3_manual_caption_20261101/annotation.json','annotation_sha256':sha(ANNOT/'annotation.json'),
        'audit_path':'grab_s3_manual_caption_20261101/audit.json','audit_sha256':sha(ANNOT/'audit.json'),'declared_source_fps':120}]})
    cfg=original_cfg;cfg.update({'dataset_root':str(HERE),'manifest':str(HERE/'manifest.json'),'stats':str(HERE/'stats.npz'),
        'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json'),'run_dir':str(HERE),'device':'cpu',
        'purpose':'unaccepted mixed-origin technical preview only',
        'manual_caption_binding':{'evidence_root':str(ROOT),'evidence_manifest':str(HERE/'manual_caption_evidence.json'),
                                  'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json')}})
    for key in ('caption_archive_binding','vertical_support_diagnostic'):
        cfg[key]['state_root']=str(HERE);cfg[key]['state_manifest']=str(HERE/'state_manifest.json')
    save(HERE/'config.json',cfg)
    out={'parent_manifest_sha256':sha(BASE/'manifest.json'),'grab_manifest_sha256':sha(GRAB/'manifest.json'),
         'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
         'stats_sha256':sha(HERE/'stats.npz'),'script_sha256':sha(Path(__file__)),
         'train_feature_frames':int(stats['frame_count']),'training_ready':False,
         'limitations':['Only five tiny candidates; source clock/identity, physical quality and license chains unresolved.',
                        'One caption is a subjective new annotation, not GRAB or HumanML source text.',
                        'Auxiliary family splits do not establish cross-family subject independence.']}
    save(HERE/'build_report.json',out);print(json.dumps(out))
if __name__=='__main__':main()
