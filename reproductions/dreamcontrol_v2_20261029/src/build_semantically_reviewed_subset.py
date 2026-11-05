"""Build an unaccepted technical subset whose captions have bounded reviews."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from training_stats import training_stats
from provenance_groups import audit_provenance

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PRE=ROOT/'four_caption_preflight_20261101'
DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main():
    torch.set_num_threads(1)
    audit=json.loads((PRE/'audit.json').read_text())
    assert sha(PRE/'manifest.json')==audit['manifest_sha256']
    assert sha(PRE/'caption_evidence_manifest.json')==audit['caption_evidence_manifest_sha256']
    cfg=json.loads((PRE/'config.json').read_text())
    records=json.loads((PRE/'manifest.json').read_text())['records']
    state_rows=json.loads(Path(cfg['state_manifest']).read_text())['records']
    kept=[r for r in records if r.get('caption_accepted') is True];ids={r['sample_id'] for r in kept}
    if ids!={'000008','011120','ACCAD_004347','MOSH_000798_crop'}:raise ValueError('reviewed subset changed')
    states=[r for r in state_rows if r['sample_id'] in ids]
    assert len(states)==len(kept)==4 and audit_provenance(kept)['source_groups_disjoint']
    train=[];spatial=[]
    for row in kept:
        if sha(DATA/row['feature_path'])!=row['feature_sha256']:raise ValueError('feature changed')
        if row['split']!='train':continue
        with np.load(DATA/row['feature_path'],allow_pickle=False) as d:x=d['features'].copy()
        train.append(('train',x))
        with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    stats=training_stats(train,min_std=.001);spatial=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],
                            spatial_mean=spatial.mean(0),spatial_std=np.maximum(spatial.std(0),.001))
    save(HERE/'manifest.json',{'records':kept,'training_ready':False})
    save(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    cfg.update({'manifest':str(HERE/'manifest.json'),'state_manifest':str(HERE/'state_manifest.json'),
                'stats':str(HERE/'stats.npz'),'run_dir':str(HERE),
                'purpose':'four reviewed captions only; sources and physical quality unaccepted'})
    save(HERE/'config.json',cfg)
    counts={s:sum(r['split']==s for r in kept) for s in ('train','val','test')}
    save(HERE/'report.json',{'parent_manifest_sha256':sha(PRE/'manifest.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
        'stats_sha256':sha(HERE/'stats.npz'),'counts':counts,'train_frames':int(stats['frame_count']),
        'excluded':[r['sample_id'] for r in records if r['sample_id'] not in ids],
        'training_ready':False,'script_sha256':sha(Path(__file__)),
        'limitations':['Semantic reviews are visual and scoped, not independent truth.',
           'Four clips and one static validation action are insufficient to establish model generalization.',
           'Source, licensing chain, real timing, ground, collision and dynamics are not fully accepted.']})
    print(json.dumps({'counts':counts,'train_frames':int(stats['frame_count']),'training_ready':False}))


if __name__=='__main__':main()
