"""Two synthetic CPU steps with real pretrained CMDM and checkpoint reload.

Only the new mask projection is trainable. The dataset and normalization are
fixtures and SMPL is stubbed. Temporary checkpoints are deliberately removed
after the test. This is not robot training or a paper result.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from robot_diffusion_stats import configure_robot_stats
from robot_motion_dataset import RobotMotionDataset
from robot_training_loop import run_training


class StubRotation2xyz:
    def __init__(self, **kwargs): self.smpl_model = torch.nn.Identity()


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/training_loop_20261030')
    checkpoint=Path('save/omnicontrol_ckpt/model_humanml3d.pt')
    with tempfile.TemporaryDirectory(prefix='synthetic-',dir=root) as folder, contextlib.redirect_stdout(sys.stderr):
        folder=Path(folder); run=folder/'run'; run.mkdir(); records=[]
        rng=np.random.default_rng(59)
        for i in range(2):
            f=folder/f'clip{i}.npz'; np.savez_compressed(f,features=rng.normal(0,.01,(12,263)).astype(np.float32))
            records.append({'sample_id':str(i),'group_id':str(i),'split':'train','caption':'a person walks',
                            'caption_source':'synthetic fixture','feature_path':f.name,
                            'feature_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        data=RobotMotionDataset(root=folder,records=records,split='train',feature_mean=np.zeros(263),
                                feature_std=np.ones(263),spatial_mean=np.zeros(66),spatial_std=np.ones(66))
        settings=SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                                 cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
        cmdm.Rotation2xyz=StubRotation2xyz
        model=cmdm.CMDM(**get_model_args(settings,None))
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        missing,unexpected=model.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing): raise ValueError('checkpoint mismatch')
        for p in model.parameters(): p.requires_grad_(False)
        attach_explicit_hint_mask(model)
        diffusion=gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine',1000,1.),
                                       model_mean_type=gd.ModelMeanType.START_X,model_var_type=gd.ModelVarType.FIXED_SMALL,
                                       loss_type=gd.LossType.MSE,dataset='humanml')
        configure_robot_stats(diffusion,np.zeros(263),np.ones(263),np.zeros(66),np.ones(66))
        optimizer=torch.optim.AdamW(model.hint_mask_embedding.parameters(),lr=1e-5)
        provenance={'dataset_manifest_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
                    'stats_sha256':hashlib.sha256(b'synthetic:zeros-ones').hexdigest(),
                    'model_init_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        config={'batch_size':2,'seed':17,'guidance_steps':1,'guidance_scale':.5,'mode':'synthetic fixture'}
        args={'run_dir':run,'provenance':provenance,'config':config,'checkpoint_every':1,
              'batch_size':2,'seed':17,'guidance_steps':1,'guidance_scale':.5,'device':torch.device('cpu')}
        metrics=[]
        first=run_training(model,diffusion,optimizer,data,max_steps=1,on_step=lambda m,c:metrics.append(m),**args)
        before=model.hint_mask_embedding.weight.detach().clone()
        second=run_training(model,diffusion,optimizer,data,max_steps=2,resume_from=run/'step_000000001.pt',
                            on_step=lambda m,c:metrics.append(m),**args)
        change=float((model.hint_mask_embedding.weight-before).abs().max())
        if first['cursor']['step']!=1 or second['cursor']['step']!=2 or change<=0:
            raise ValueError('resume witness failed')
        sizes=[(run/f'step_{i:09d}.pt').stat().st_size for i in (1,2)]
        result={'first':first,'second':second,'losses':[m['loss'] for m in metrics],
                'mask_weight_change_after_reload':change,'checkpoint_sizes':sizes,
                'warning':'Synthetic CPU test with temporary checkpoints, only mask projection trained, human pretrained CMDM and CLIP, SMPL stubbed; not robot training or physics validation.'}
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__': main()
