"""Verify explicit-mask extension on the real pretrained CMDM in CPU mode.

Loads real CLIP and denoiser weights, stubs SMPL; uses synthetic motion/hints.
It is not training and does not establish robot motion quality.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask


class StubRotation2xyz:
    def __init__(self, **kwargs):
        self.smpl_model = torch.nn.Identity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(0)
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with contextlib.redirect_stdout(sys.stderr):
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(settings, None))
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('checkpoint not compatible')
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        x = torch.zeros((1, 263, 1, 20)); t = torch.tensor([500]); hint = torch.zeros((1, 20, 66))
        text = ['a person walks forward']; hint[0, 10, 60] = .2
        with torch.no_grad():
            old = model(x, t, y={'text': text, 'hint': hint})
        attach_explicit_hint_mask(model)
        mask = torch.zeros((1, 20, 22), dtype=torch.bool); mask[0, 10, 20] = True
        with torch.no_grad():
            new = model(x, t, y={'text': text, 'hint': hint, 'hint_mask': mask})
        compatibility = float((old-new).abs().max())
        if compatibility > 1e-6:
            raise ValueError(f'zero-init changed an ordinary hint: {compatibility}')
        zero_hint = torch.zeros_like(hint)
        output = model(x, t, y={'text': text, 'hint': zero_hint, 'hint_mask': mask})
        output.square().mean().backward()
        gradient = model.hint_mask_embedding.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() <= 0:
            raise ValueError('zero target did not yield finite mask-branch gradient')
        with torch.no_grad():
            model.hint_mask_embedding.weight.fill_(.01)
            active = model(x, t, y={'text': text, 'hint': zero_hint, 'hint_mask': mask})
            inactive = model(x, t, y={'text': text, 'hint': zero_hint, 'hint_mask': torch.zeros_like(mask)})
        delta = float(torch.linalg.vector_norm(active-inactive))
        if delta <= 0:
            raise ValueError('explicit zero-valued hint not represented')
    print(json.dumps({'original_checkpoint_keys': len(state), 'max_pretrained_compatibility_error': compatibility,
                      'mask_weight_grad_l1': float(gradient.abs().sum()), 'active_zero_target_difference_l2': delta,
                      'note': 'Synthetic forward/backward only; zero-init mask projection needs real robot training. SMPL stubbed.'}))


if __name__ == '__main__':
    main()
