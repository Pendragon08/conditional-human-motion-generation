"""CPU loader of the published60->20Hz motion/text profile for future prior training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path.insert(0,str(T))
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self,root,split,stats):
        self.root=Path(root);self.rows=[r for r in json.loads((self.root/'manifest.json').read_text())['records'] if r['split']==split]
        self.stats=stats
        if not self.rows:raise ValueError('empty dataset split')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];p=(self.root/row['feature_path']).resolve()
        if not p.is_relative_to(self.root.resolve()) or sha(p)!=row['feature_sha256']:raise ValueError('dataset artifact changed')
        with np.load(p,allow_pickle=False) as z:
            raw=torch.tensor(z['features'],dtype=torch.float32)
            state=torch.tensor(z['qpos29_wxyz_xyz'],dtype=torch.float32)
        points=recover_from_ric(raw,22)
        return {'row':row,'motion':(raw-self.stats['feature_mean'])/self.stats['feature_std'],
                'points':points,'captions':[x['caption'] for x in row['captions']],
                'qpos29_wxyz_xyz':state,'length':len(raw)}


def collate_motion_text(items):
    maximum=max(x['length'] for x in items);n=len(items)
    motion=torch.zeros(n,263,1,maximum);valid=torch.zeros(n,1,1,maximum,dtype=torch.bool)
    points=torch.zeros(n,maximum,22,3);texts=[]
    for i,x in enumerate(items):
        length=x['length'];motion[i,:,0,:length]=x['motion'].T;valid[i,0,0,:length]=True
        points[i,:length]=x['points'];texts.append(x['captions'][0])
    return {'motion':motion,'valid_mask':valid,'points':points,'text':texts,
            'lengths':torch.tensor([x['length'] for x in items]),'full29_states':[x['qpos29_wxyz_xyz'] for x in items]}


def prepare_spatial_stats(root,output_dir):
    root=Path(root);output_dir=Path(output_dir);output_dir.mkdir(exist_ok=False)
    rows=json.loads((root/'manifest.json').read_text())['records']
    with np.load(root/'stats.npz',allow_pickle=False) as z:
        feature_mean=z['feature_mean'].copy();feature_std=z['feature_std'].copy()
    count=0;s=np.zeros(66);sq=np.zeros(66)
    for row in rows:
        if row['split']!='train':continue
        file=root/row['feature_path'];assert sha(file)==row['feature_sha256']
        with np.load(file,allow_pickle=False) as z:raw=torch.tensor(z['features'],dtype=torch.float32)
        points=recover_from_ric(raw,22).numpy().reshape(-1,66).astype(float)
        assert np.isfinite(points).all()
        count+=len(points);s+=points.sum(0);sq+=np.square(points).sum(0)
    mean=s/count;std=np.maximum(np.sqrt(np.maximum(sq/count-mean**2,0.)),1e-6)
    path=output_dir/'combined_stats.npz'
    with path.open('xb') as f:np.savez(f,feature_mean=feature_mean,feature_std=feature_std,
        spatial_mean=mean.astype(np.float32),spatial_std=std.astype(np.float32),training_feature_frames=count)
    with np.load(path,allow_pickle=False) as z:stats={k:torch.tensor(z[k]) for k in ['feature_mean','feature_std','spatial_mean','spatial_std']}
    witnesses={}
    for split in ['train','val','test']:
        dataset=PairedDataset(root,split,stats)
        indices=[0,len(dataset)//3,2*len(dataset)//3,len(dataset)-1]
        items=[dataset[i] for i in indices];batch=collate_motion_text(items)
        assert batch['motion'].shape[:3]==(4,263,1) and batch['points'].shape[2:]==(22,3)
        assert batch['valid_mask'].sum()==sum(x['length'] for x in items)
        assert torch.isfinite(batch['motion']).all() and all(batch['text'])
        spatial=(batch['points'].flatten(-2)-stats['spatial_mean'])/stats['spatial_std']
        assert torch.isfinite(spatial).all()
        assert all(x.shape[1]==36 for x in batch['full29_states'])
        witnesses[split]={'rows':len(dataset),'sampled_ids':[x['row']['sample_id'] for x in items],
            'batch_shape':list(batch['motion'].shape),'valid_feature_frames':int(batch['valid_mask'].sum()),'motion_text_and29states_loaded':True}
    result={'status':'combined_stats_and_three_split_loader_passed','training_feature_frames':count,
        'combined_stats_sha256':sha(path),'base_feature_stats_sha256':sha(root/'stats.npz'),
        'manifest_sha256':sha(root/'manifest.json'),'loader_witnesses':witnesses,
        'temporal_profile':'publisher_nominal60_to20Hz','training_started':False,
        'note':'no CLIP/model inference or real-time-clock proof claimed; raw263D normalized only using this profile train split'}
    with (output_dir/'report.json').open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);print(json.dumps(prepare_spatial_stats(a.dataset,a.output_dir)),flush=True)
