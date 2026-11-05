"""Measure synthetic checkpoint serialization in a discard-only sink.

Constructs the existing old CMDM/CLIP configuration on CPU. AdamW moment
buffers are allocated as zeros; no training step, dataset, GPU or output
checkpoint is used. SMPL helper is stubbed and CUDA RNG/config overhead may
vary, so this is a lower-bound planning measurement, not a safe hard limit.
"""
import hashlib,importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch,clip
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INIT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/save/omnicontrol_ckpt/model_humanml3d.pt')
CLIP=Path('/home/ouyanghao/.cache/clip/ViT-B-32.pt')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
class StubRotation2xyz:
    def __init__(self,**kwargs):self.smpl_model=torch.nn.Identity()
class Counter:
    def __init__(self):self.n=0
    def write(self,data):self.n+=len(data);return len(data)
    def flush(self):pass
    def tell(self):return self.n

def main():
    torch.set_num_threads(1)
    clip_impl=importlib.import_module('clip.clip')
    if sha(CLIP)!=clip_impl._MODELS['ViT-B/32'].split('/')[-2]:raise ValueError('CLIP cache mismatch')
    source_sha,clip_sha=sha(INIT),sha(CLIP)
    args=SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                         cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
    cmdm.Rotation2xyz=StubRotation2xyz
    with patch.object(clip_impl,'_download',return_value=str(CLIP)):
        model=cmdm.CMDM(**get_model_args(args,None))
    weights=torch.load(INIT,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if unexpected or any(not k.startswith('clip_model.') for k in missing):raise ValueError('checkpoint mismatch')
    attach_explicit_hint_mask(model)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-5,weight_decay=0.)
    for p in trainable:
        optimizer.state[p]={'step':torch.tensor(1.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    rng={'python':None,'numpy':list(np.random.get_state()[1][:624].astype(int)),'torch':torch.get_rng_state(),'cuda':[]}
    payload={'version':1,'provenance':{'dataset_manifest_sha256':'0'*64,'stats_sha256':'0'*64,'model_init_sha256':source_sha},
             'config':{'example':'synthetic_measurement'},'cursor':{'step':1,'epoch':0,'next_batch':0},
             'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng}
    sink=Counter();torch.save(payload,sink)
    result={'checkpoint_init_sha256':source_sha,'clip_sha256':clip_sha,'script_sha256':sha(Path(__file__)),
            'serialized_bytes_discarded':sink.n,'trainable_parameter_elements':sum(p.numel() for p in trainable),
            'all_parameter_elements':sum(p.numel() for p in model.parameters()),'optimizer_state_entries':len(optimizer.state),
            'model_state_keys':len(model.state_dict()),'training_ready':False,
            'limitations':['Real CPU CMDM and CLIP, but SMPL helper stubbed and CUDA RNG omitted.',
                           'All synthetic Adam states allocated with zeros; no optimizer step or checkpoint written.',
                           'Config and device overhead may differ; measured bytes are not a safe upper bound.',
                           'Training data and checkpoint space remain unaccepted.']}
    out=Path(__file__).parent/'report.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
