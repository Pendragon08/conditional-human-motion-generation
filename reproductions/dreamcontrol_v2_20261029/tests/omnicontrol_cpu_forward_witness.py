"""CPU synthetic forward check of the existing pretrained human OmniControl.

Uses the existing CLIP and loads the real checkpoint. SMPL is stubbed because
this HumanML feature forward pass does not call it and the local legacy SMPL
dependency is unavailable. This is not DreamControl-v2 training or quality.
"""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

from utils.model_util import get_model_args
import model.cmdm as cmdm


class StubRotation2xyz:
    def __init__(self, **kwargs):
        self.smpl_model = torch.nn.Identity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with contextlib.redirect_stdout(sys.stderr):
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(settings, None))
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not name.startswith('clip_model.') for name in missing):
            raise ValueError(f'unexpected model keys: {unexpected}; non-CLIP missing: {missing}')
        model.eval()
        x = torch.zeros((1, 263, 1, 20), dtype=torch.float32)
        timestep = torch.tensor([500], dtype=torch.long)
        hint = torch.zeros((1, 20, 66), dtype=torch.float32)
        text = ['a person walks forward']
        with torch.inference_mode():
            baseline = model(x, timestep, y={'text': text, 'hint': hint})
            hint[0, 10, 60] = .2
            controlled = model(x, timestep, y={'text': text, 'hint': hint})
    delta = (baseline - controlled).abs()
    h = hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    print(json.dumps({'checkpoint_sha256': h.hexdigest(), 'output_shape': list(controlled.shape),
                      'baseline_finite': bool(torch.isfinite(baseline).all()),
                      'controlled_finite': bool(torch.isfinite(controlled).all()),
                      'spatial_hint_change_l2': float(torch.linalg.vector_norm(delta)),
                      'missing_clip_keys': len(missing), 'unexpected_keys': unexpected,
                      'caveat': 'Synthetic CPU forward of a human pretrained model with real CLIP and stubbed SMPL; not a robot model or paper result.'}))


if __name__ == '__main__':
    main()
