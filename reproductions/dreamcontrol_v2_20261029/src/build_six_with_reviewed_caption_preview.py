"""Extend the five-case technical preview with one semantically reviewed SSM clip.

This diagnostic set is not approved for training; physical, timing, source and
licensing evidence remain unresolved. No original data or statistics are changed.
"""
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
BASE=ROOT/'five_conservative_preview_20261031'
INPUT=ROOT/'ssm_anchored_20hz_candidates_20261031'
SEM=ROOT/'ssm_011120_semantic_review_20261101'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def copy(source,dest,expected):
    if sha(source)!=expected:raise ValueError('input changed')
    with source.open('rb') as a,dest.open('xb') as b:shutil.copyfileobj(a,b)
    if sha(dest)!=expected:raise ValueError('copy mismatch')


def main():
    torch.set_num_threads(1)
    parent=json.loads((BASE/'report.json').read_text())
    if sha(BASE/'manifest.json')!=parent['manifest_sha256'] or sha(BASE/'state_manifest.json')!=parent['state_manifest_sha256']:
        raise ValueError('parent manifest changed')
    previous=json.loads((BASE/'manifest.json').read_text())['records']
    old_states=json.loads((BASE/'state_manifest.json').read_text())['records']
    receipt=json.loads((SEM/'semantic_review.json').read_text())
    check=json.loads((SEM/'independent_audit.json').read_text())
    human=json.loads((ROOT/'ssm_rawfps_human_batch_20261031/011120/human_report.json').read_text())
    if sha(SEM/'semantic_review.json')!=check['receipt_sha256'] or not receipt['manual_semantic_assessment']['semantic_accepted_for_exported_samples']:
        raise ValueError('semantic review evidence missing')
    if sha(INPUT/'011120.npz')!=receipt['candidate_sha256'] or receipt['caption_member_sha256']!=human['caption_member_sha256']:
        raise ValueError('review does not bind data or caption')
    source=human['source_path']; group=human['source_parent']; sid='011120'
    if sid in {r['sample_id'] for r in previous} or any(group in r['provenance_keys'].get('source_directory',[]) for r in previous):
        raise ValueError('source group already present; resolve explicitly')
    name=sid+'.npz'; digest=receipt['candidate_sha256']
    record={'sample_id':sid,'source_id':source,'group_id':group,
            'provenance_keys':{'source_path':[source],'source_directory':[group]},
            'split':'train','caption':receipt['caption'],'caption_source':'HumanML texts.zip: texts/011120.txt',
            'caption_accepted':True,'caption_review_sha256':sha(SEM/'semantic_review.json'),
            'feature_path':name,'feature_sha256':digest,'fps':20,'training_ready':False,
            'auxiliary_split_note':'diagnostic group holdout only; source/subject identity not proven'}
    state={'sample_id':sid,'state_path':name,'state_sha256':digest,'feature_sha256':digest,
           'foot_velocity_squared_threshold':.002}
    for row in previous:copy(BASE/row['feature_path'],HERE/row['feature_path'],row['feature_sha256'])
    for row in old_states:copy(BASE/row['state_path'],HERE/row['state_path'],row['state_sha256'])
    copy(INPUT/name,HERE/name,digest)
    records=previous+[record]; states=old_states+[state]
    groups=audit_provenance(records)
    if not groups['source_groups_disjoint']:raise ValueError('known provenance crosses split')
    train=[]; spatial=[]
    for row in records:
        if row['split']!='train':continue
        with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
        train.append(('train',x))
        with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    stats=training_stats(train,min_std=.001); spatial=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],
                            spatial_mean=spatial.mean(0),spatial_std=np.maximum(spatial.std(0),.001))
    save(HERE/'manifest.json',{'records':records,'training_ready':False})
    save(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    config=json.loads((BASE/'preflight_config.json').read_text())
    config.update({'dataset_root':str(HERE),'manifest':str(HERE/'manifest.json'),'stats':str(HERE/'stats.npz'),
                   'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json'),'run_dir':str(HERE),
                   'device':'cpu','purpose':'unaccepted six-record technical preview; one bounded caption review'})
    save(HERE/'preflight_config.json',config)
    save(HERE/'report.json',{'parent_manifest_sha256':sha(BASE/'manifest.json'),
        'semantic_receipt_sha256':sha(SEM/'semantic_review.json'),'semantic_receipt_audit_sha256':sha(SEM/'independent_audit.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
        'stats_sha256':sha(HERE/'stats.npz'),'script_sha256':sha(Path(__file__)),
        'train_frames':int(stats['frame_count']),'training_ready':False,
        'limitations':['Auxiliary source-path grouping is not proof of recording or subject independence.',
          'One bounded visual semantic judgment; the other five captions remain unaccepted.',
          'True timing, source identity, licensing chain and physical quality remain unaccepted.',
          'A tiny and static validation split cannot establish model generalization.']})
    print(json.dumps({'cases':len(records),'train_frames':int(stats['frame_count']),'training_ready':False}))


if __name__=='__main__':main()
