"""Single synthetic CPU integration step; no checkpoint or dataset is saved.

Real pretrained OmniControl and CLIP are used; SMPL is stubbed. Random input,
caption and statistics are fixtures and do not represent robot training.
"""
import contextlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from robot_diffusion_stats import configure_robot_stats
from robot_motion_dataset import RobotMotionDataset, collate_robot_motion
from robot_spatial_guidance import guide


class StubRotation2xyz:
    def __init__(self, **kwargs):
        self.smpl_model = torch.nn.Identity()


def main():
    torch.set_num_threads(1); torch.manual_seed(11)
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with tempfile.TemporaryDirectory(prefix='dcv2_fixture_') as directory, contextlib.redirect_stdout(sys.stderr):
        root = Path(directory); feature_file = root/'synthetic.npz'
        np.savez_compressed(feature_file, features=np.random.default_rng(11).normal(0, .01, (20, 263)).astype(np.float32))
        record = {'sample_id':'fixture', 'group_id':'fixture', 'split':'train', 'caption':'a person walks forward',
                  'caption_source':'synthetic test only', 'feature_path':feature_file.name,
                  'feature_sha256':hashlib.sha256(feature_file.read_bytes()).hexdigest()}
        data = RobotMotionDataset(root=root, records=[record], split='train', feature_mean=np.zeros(263),
                                  feature_std=np.ones(263), spatial_mean=np.zeros(66), spatial_std=np.ones(66))
        clean, cond = collate_robot_motion([data[0]])
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(settings, None))
        checkpoint = Path('save/omnicontrol_ckpt/model_humanml3d.pt')
        missing, unexpected = model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('checkpoint mismatch')
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        attach_explicit_hint_mask(model)
        diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine', 1000, 1.),
                                        model_mean_type=gd.ModelMeanType.START_X, model_var_type=gd.ModelVarType.FIXED_SMALL,
                                        loss_type=gd.LossType.MSE, dataset='humanml')
        configure_robot_stats(diffusion, np.zeros(263), np.ones(263), np.zeros(66), np.ones(66))
        times = torch.tensor([500], dtype=torch.long)
        noisy = diffusion.q_sample(clean, times, noise=torch.randn_like(clean))
        guided = guide(diffusion, noisy, times, cond['y']['hint'], cond['y']['hint_mask'],
                       step_size=.5, steps=2, valid_frames=cond['y']['mask'][:, 0, 0, :])
        optimizer = torch.optim.AdamW(model.hint_mask_embedding.parameters(), lr=1e-5)
        before = model.hint_mask_embedding.weight.detach().clone()
        prediction = model(guided, times, y=cond['y'])
        valid = cond['y']['mask'].expand_as(prediction)
        loss = (prediction-clean).square().masked_select(valid).mean()
        loss.backward()
        grad = model.hint_mask_embedding.weight.grad
        if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
            raise ValueError('smoke step produced invalid loss or mask gradient')
        optimizer.step()
        changed = float((model.hint_mask_embedding.weight-before).abs().max())
        if changed <= 0:
            raise ValueError('synthetic optimizer step changed no parameter')
        result = {'fixture_frames': 20, 'input_shape': list(clean.shape), 'output_shape': list(prediction.shape),
                  'loss': float(loss), 'mask_weight_gradient_l1': float(grad.abs().sum()), 'max_weight_change': changed,
                  'caveat': 'One synthetic CPU step; only experimental mask projection updated in memory; SMPL stubbed; no real training or checkpoint.'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
