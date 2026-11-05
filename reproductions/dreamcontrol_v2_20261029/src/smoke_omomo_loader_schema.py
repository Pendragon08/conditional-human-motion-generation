"""Exercise the existing loader on an unaccepted OMOMO schema preview.

CPU integration check only: no model, optimizer, GPU or training. Captions
remain unaccepted and source/time/physics remain unresolved.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from robot_motion_dataset import RobotMotionDataset,collate_robot_motion

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'omomo_training_schema_preview_20261101'
STATS=ROOT/'omomo_mixed_encoding_preview_20261101/stats.npz'
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1);torch.manual_seed(53)
    report=json.loads((INPUT/'report.json').read_text()); mh=digest(INPUT/'manifest.json');sh=digest(STATS)
    if mh!=report['manifest_sha256'] or sh!=report['source_stats_sha256']:raise ValueError('input hash mismatch')
    records=json.loads((INPUT/'manifest.json').read_text())['records']
    if any(r['caption_accepted'] is not False or r['training_ready'] is not False for r in records):raise ValueError('unexpected accepted sample')
    with np.load(STATS,allow_pickle=False) as f:stats={k:f[k].copy() for k in ('feature_mean','feature_std','spatial_mean','spatial_std')}
    cases=[]
    for split in ('train','val','test'):
        dataset=RobotMotionDataset(root=ROOT,records=records,split=split,**stats)
        item=dataset[0];motion,cond=collate_robot_motion([item]); n=len(item['motion'])
        if motion.shape!=(1,263,1,n) or cond['y']['mask'].shape!=(1,1,1,n) or not cond['y']['mask'].all():
            raise ValueError('loader output shape or mask mismatch')
        if not torch.isfinite(motion).all() or not torch.isfinite(cond['y']['hint']).all():raise ValueError('non-finite loader output')
        cases.append({'split':split,'dataset_records':len(dataset),'sample_id':item['sample_id'],
                      'feature_frames':n,'motion_shape':list(motion.shape),'hint_shape':list(cond['y']['hint'].shape),
                      'selected_hint_points':int(cond['y']['hint_mask'].sum())})
    if digest(INPUT/'manifest.json')!=mh or digest(STATS)!=sh:raise RuntimeError('input changed during smoke')
    result={'manifest_sha256':mh,'stats_sha256':sh,'script_sha256':digest(Path(__file__)),'cases':cases,'training_ready':False,
            'limitations':['Loader verifies file hashes and one sample per split, not label truth or physical data quality.',
                           'Random spatial hint policy and old decoder are approximations, not author v2 validation.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()
