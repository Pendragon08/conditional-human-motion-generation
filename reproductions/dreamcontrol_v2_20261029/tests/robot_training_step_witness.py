"""Synthetic CPU integration check of one optimizer step; saves no weights."""
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from robot_diffusion_stats import configure_robot_stats
from robot_training_step import train_step


class StubRotation2xyz:
    def __init__(self, **kwargs):
        self.smpl_model = torch.nn.Identity()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(59)
    with contextlib.redirect_stdout(sys.stderr):
        args = SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                               cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(args,None))
        state = torch.load(Path('save/omnicontrol_ckpt/model_humanml3d.pt'),map_location='cpu',weights_only=True)
        missing, unexpected = model.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('checkpoint mismatch')
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        attach_explicit_hint_mask(model)
        diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine',1000,1.),
                                        model_mean_type=gd.ModelMeanType.START_X,
                                        model_var_type=gd.ModelVarType.FIXED_SMALL,
                                        loss_type=gd.LossType.MSE,dataset='humanml')
        configure_robot_stats(diffusion,np.zeros(263),np.ones(263),np.zeros(66),np.ones(66))
        x = torch.randn(2,263,1,12)*.01
        mask = torch.zeros(2,12,22,dtype=torch.bool)
        mask[0,5,20]=True; mask[1,7,21]=True
        hint = torch.zeros(2,12,66); hint[0,5,60:63]=torch.tensor([.2,.1,.3]); hint[1,7,63:66]=torch.tensor([-.1,.2,.1])
        cond = {'y':{'mask':torch.ones(2,1,1,12,dtype=torch.bool),'hint':hint,'hint_mask':mask,
                     'text':['a person walks','a person reaches']}}
        optimizer = torch.optim.AdamW(model.hint_mask_embedding.parameters(),lr=1e-5)
        before = model.hint_mask_embedding.weight.detach().clone()
        result = train_step(model,diffusion,optimizer,x,cond,guidance_steps=1,guidance_scale=.5,
                            timesteps=torch.tensor([450,550]))
        delta = float((model.hint_mask_embedding.weight-before).abs().max())
        if not delta > 0: raise ValueError('mask projection did not update')
        result.update({'mask_weight_change':delta,
                       'warning':'Synthetic CPU one-step check using pretrained human model and CLIP, SMPL stubbed. Only mask projection trained; no checkpoint, robot model training or physics validation.'})
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
