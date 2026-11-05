"""Create conservative source-family holdouts for a six-clip diagnostic preview.

Grouping all SSM recordings together is a deliberate project-side precaution,
not an official HumanML partition or proof of cross-family independence.
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
BASE=ROOT/'six_reviewed_caption_preview_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def copy(source,dest,expected):
    if sha(source)!=expected:raise ValueError('input changed')
    with source.open('rb') as a,dest.open('xb') as b:shutil.copyfileobj(a,b)
    if sha(dest)!=expected:raise ValueError('copy changed')


def main():
    torch.set_num_threads(1)
    parent=json.loads((BASE/'report.json').read_text()); audit=json.loads((BASE/'independent_audit.json').read_text())
    assert sha(BASE/'manifest.json')==parent['manifest_sha256']==audit['manifest_sha256']
    assert sha(BASE/'state_manifest.json')==parent['state_manifest_sha256']
    records=json.loads((BASE/'manifest.json').read_text())['records']
    states=json.loads((BASE/'state_manifest.json').read_text())['records']
    if len(records)!=6 or len(states)!=6:raise ValueError('unexpected preview')
    out=[]; train=[]; spatial=[]
    for row in records:
        path=row['source_id'].removeprefix('./')
        if path.startswith('pose_data/SSM_synced/'):
            family,split='SSM_synced','train'
        elif path.startswith('pose_data/ACCAD/'):
            family,split='ACCAD','val'
        elif path.startswith('pose_data/MPI_mosh/'):
            family,split='MPI_mosh','test'
        else:raise ValueError('unsupported source family '+path)
        copy(BASE/row['feature_path'],HERE/row['feature_path'],row['feature_sha256'])
        rec=dict(row);rec['provenance_keys']={k:list(v) for k,v in row['provenance_keys'].items()}
        rec['provenance_keys']['source_family']=[family]
        rec['group_id']='conservative-family:'+family; rec['split']=split
        rec['auxiliary_split_note']='source-family holdout; not official split and not proof of identity independence'
        out.append(rec)
        if split=='train':
            with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
            train.append(('train',x))
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(float))
    for row in states:
        dest=HERE/row['state_path']
        if dest.exists():
            if sha(dest)!=row['state_sha256'] or sha(BASE/row['state_path'])!=row['state_sha256']:
                raise ValueError('shared feature/state path has inconsistent hash')
        else:copy(BASE/row['state_path'],dest,row['state_sha256'])
    if not audit_provenance(out)['source_groups_disjoint']:raise ValueError('provenance overlaps splits')
    stats=training_stats(train,min_std=.001); pts=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],
                            spatial_mean=pts.mean(0),spatial_std=np.maximum(pts.std(0),.001))
    save(HERE/'manifest.json',{'records':out,'training_ready':False})
    save(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    config=json.loads((BASE/'preflight_config.json').read_text())
    config.update({'dataset_root':str(HERE),'manifest':str(HERE/'manifest.json'),'stats':str(HERE/'stats.npz'),
                   'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json'),
                   'run_dir':str(HERE),'device':'cpu','purpose':'unaccepted conservative source-family review'})
    save(HERE/'preflight_config.json',config)
    counts={s:sum(r['split']==s for r in out) for s in ('train','val','test')}
    save(HERE/'report.json',{'parent_manifest_sha256':sha(BASE/'manifest.json'),'parent_audit_sha256':sha(BASE/'independent_audit.json'),
         'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
         'stats_sha256':sha(HERE/'stats.npz'),'script_sha256':sha(Path(__file__)),
         'counts':counts,'train_frames':int(stats['frame_count']),'training_ready':False,
         'limitations':['Coarse family split is not official HumanML and does not prove cross-family subject independence.',
         'A single static validation action and a single held-out test action are insufficient for evaluation.',
         'Timing, source identity, applicable license chain, caption and physical quality are not all accepted.']})
    print(json.dumps({'counts':counts,'train_frames':int(stats['frame_count']),'training_ready':False}))


if __name__=='__main__':main()
