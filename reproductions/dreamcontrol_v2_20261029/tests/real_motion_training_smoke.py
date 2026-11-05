"""CPU-only two-clip end-to-end data/model plumbing check, no saved weights.

Input is trusted existing 29-DoF corpus with original captions. Derived clips
are temporary and not physically validated; statistics come from these two
clips only. This does not constitute robot training or paper reproduction.
"""
import contextlib
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from motion_bridge import prepare_root_fitted_clip
from robot_diffusion_stats import configure_robot_stats
from robot_motion_dataset import RobotMotionDataset, collate_robot_motion
from robot_spatial_guidance import guide
from training_stats import training_stats
from waist_lock_root_ik import fit_root_and_waist


class StubRotation2xyz:
    def __init__(self, **kwargs):
        self.smpl_model = torch.nn.Identity()


def main():
    torch.set_num_threads(1); torch.manual_seed(37)
    source = Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl')
    repo = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets = repo/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    with source.open('rb') as f:
        corpus = pickle.load(f)['data_dict']  # Trusted local project file only.
    fk = URDFFK(u27)
    clips, source_rows = [], []
    for key in ['000000', '000005']:
        original = corpus[key]
        motion = np.asarray(original['motion'][::3])
        fit = fit_root_and_waist(motion, u29, u27, steps=120, foot_position_weight=2., contact_velocity_weight=10.)
        prepared = prepare_root_fitted_clip(motion, fit, fk)
        features = encode(prepared['points_z_up_world'], foot_velocity_squared_threshold=.002)
        with torch.no_grad():
            decoded = recover_from_ric(torch.from_numpy(features), 22).numpy()
        expected = canonicalize_z_up(prepared['points_z_up_world'])[:-1]
        reconstruction_max = float(np.linalg.norm(decoded-expected, axis=-1).max())
        if reconstruction_max > 1e-4:
            raise ValueError(f'feature reconstruction mismatch for {key}: {reconstruction_max}')
        captions = [item['caption'] for item in original['text'] if isinstance(item.get('caption'), str) and item['caption'].strip()]
        if not captions:
            raise ValueError(f'no source caption for {key}')
        clips.append((features, decoded, captions[0], key))
        source_rows.append({'id':key, 'input_frames':len(motion), 'feature_frames':len(features),
                            'reconstruction_max_m':reconstruction_max,
                            'fit_mean_keypoint_error_m':fit['final_mean_keypoint_error_m'],
                            'fit_max_keypoint_error_m':fit['final_max_keypoint_error_m']})
    stats = training_stats([('train', x[0]) for x in clips], min_std=1e-3)
    positions = np.concatenate([x[1] for x in clips], axis=0)
    raw_mean = positions.mean(axis=0); raw_std = np.maximum(positions.std(axis=0), 1e-3)
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with tempfile.TemporaryDirectory(prefix='dcv2_real_smoke_') as directory, contextlib.redirect_stdout(sys.stderr):
        root=Path(directory); records=[]
        for features, _, caption, key in clips:
            file = root/(key+'.npz'); np.savez_compressed(file, features=features)
            records.append({'sample_id':key, 'group_id':key, 'split':'train', 'caption':caption,
                            'caption_source':'existing trusted AMASS/G1 corpus; group independence not verified',
                            'feature_path':file.name, 'feature_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        data = RobotMotionDataset(root=root, records=records, split='train', feature_mean=stats['mean'],
                                  feature_std=stats['std'], spatial_mean=raw_mean, spatial_std=raw_std)
        clean, cond = collate_robot_motion([data[0], data[1]])
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(settings, None))
        state = torch.load('save/omnicontrol_ckpt/model_humanml3d.pt', map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('checkpoint mismatch')
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        attach_explicit_hint_mask(model)
        diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine', 1000, 1.),
                                         model_mean_type=gd.ModelMeanType.START_X,
                                         model_var_type=gd.ModelVarType.FIXED_SMALL, loss_type=gd.LossType.MSE,
                                         dataset='humanml')
        configure_robot_stats(diffusion, stats['mean'], stats['std'], raw_mean, raw_std)
        times = torch.tensor([450, 550], dtype=torch.long)
        noisy = diffusion.q_sample(clean, times, noise=torch.randn_like(clean))
        guided = guide(diffusion, noisy, times, cond['y']['hint'], cond['y']['hint_mask'],
                       step_size=.5, steps=2, valid_frames=cond['y']['mask'][:, 0, 0, :])
        optimizer = torch.optim.AdamW(model.hint_mask_embedding.parameters(), lr=1e-5)
        before = model.hint_mask_embedding.weight.detach().clone()
        prediction = model(guided, times, y=cond['y'])
        valid = cond['y']['mask'].expand_as(prediction)
        loss = (prediction-clean).square().masked_select(valid).mean()
        loss.backward(); grad = model.hint_mask_embedding.weight.grad
        if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
            raise ValueError('invalid smoke-test loss or gradient')
        optimizer.step(); delta=float((model.hint_mask_embedding.weight-before).abs().max())
        if delta <= 0: raise ValueError('mask projection was not updated')
        result = {'source_samples':source_rows, 'batch_shape':list(clean.shape), 'loss':float(loss),
                  'train_frame_count':stats['frame_count'], 'clamped_feature_dims':len(stats['clamped_channels']),
                  'mask_gradient_l1':float(grad.abs().sum()), 'max_mask_weight_delta':delta,
                  'caveat':'Two real clips and original captions, no physics validation; one in-memory CPU step; SMPL stubbed; no model or dataset saved.'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
