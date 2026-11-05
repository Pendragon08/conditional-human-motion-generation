"""CPU-only synthetic sampling through the old pretrained CMDM and real CLIP.

This does not train, create a checkpoint, use accepted robot data, or validate
DreamControl-v2 fidelity.  SMPL is stubbed because only 263D decoding is used.
"""
import hashlib,importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import clip
import numpy as np
import torch
from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from robot_diffusion_stats import configure_robot_stats
from robot_prior_sampling import sample_robot_prior

HERE=Path(__file__).resolve().parent
CHECKPOINT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/save/omnicontrol_ckpt/model_humanml3d.pt')
CLIP=Path('/home/ouyanghao/.cache/clip/ViT-B-32.pt')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


class StubRotation2xyz:
    def __init__(self,**kwargs):self.smpl_model=torch.nn.Identity()


def main():
    torch.set_num_threads(1)
    # The cache must already exist and match the upstream URL; never download.
    clip_impl=importlib.import_module('clip.clip')
    expected_clip=clip_impl._MODELS['ViT-B/32'].split('/')[-2]
    if not CLIP.is_file() or sha(CLIP)!=expected_clip:raise ValueError('cached CLIP absent or mismatched')
    checkpoint_sha=sha(CHECKPOINT);clip_sha=sha(CLIP)
    args=SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                         cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
    cmdm.Rotation2xyz=StubRotation2xyz
    with patch.object(clip_impl,'_download',return_value=str(CLIP)):
        model=cmdm.CMDM(**get_model_args(args,None))
    weights=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if unexpected or any(not key.startswith('clip_model.') for key in missing):raise ValueError('human checkpoint mismatch')
    attach_explicit_hint_mask(model);model.eval()
    def fake_stats(path):
        n=66 if 'raw' in str(path).lower() else 263
        return np.ones(n,dtype=np.float32) if 'Std' in str(path) else np.zeros(n,dtype=np.float32)
    with patch.object(gd.np,'load',side_effect=fake_stats):
        diffusion=gd.GaussianDiffusion(betas=np.linspace(.001,.02,4),model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,loss_type=gd.LossType.MSE,dataset='humanml')
    configure_robot_stats(diffusion,np.zeros(263),np.ones(263),np.zeros(66),np.ones(66))
    targets=torch.zeros((1,10,22,3));mask=torch.zeros((1,10,22),dtype=torch.bool)
    targets[0,5,20]=torch.tensor([.2,0.,.1]);mask[0,5,20]=True
    kwargs={'texts':['a person reaches forward'],'targets':targets,'target_mask':mask,
            'cfg_scale':2.5,'spatial_step_size':.05,'spatial_steps':1,'seed':772}
    first=sample_robot_prior(model,diffusion,**kwargs)
    second=sample_robot_prior(model,diffusion,**kwargs)
    delta=float((first['normalized_features']-second['normalized_features']).abs().max())
    if delta!=0.:raise ValueError('same-seed CPU output differs')
    if sha(CHECKPOINT)!=checkpoint_sha or sha(CLIP)!=clip_sha:raise RuntimeError('pretrained inputs changed')
    features=first['normalized_features'].cpu().numpy();points=first['canonical_points'].cpu().numpy()
    result={'checkpoint_sha256':checkpoint_sha,'clip_sha256':clip_sha,'script_sha256':sha(Path(__file__)),
            'sampler_sha256':sha(HERE/'robot_prior_sampling.py'),'explicit_mask_sha256':sha(HERE/'explicit_hint_control.py'),
            'sample_shape':list(features.shape),'point_shape':list(points.shape),'sample_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
            'same_seed_max_abs_difference':delta,'metadata':first['metadata'],'training_ready':False,
            'limitations':['Four synthetic diffusion steps with old human checkpoint, not a v2 or robot-trained model.',
                           'CLIP is real, SMPL is stubbed, statistics are synthetic; no checkpoint saved.',
                           'No 263D-to-27DoF postoptimization, physics or v2 quality validation.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
