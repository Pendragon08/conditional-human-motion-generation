"""CPU-only shape audit for the existing OmniControl human checkpoint.

No model inference or training. CLIP and SMPL are stubbed only to inspect
denoiser/control parameter shapes without downloading models or loading body
assets; their functionality and weights are NOT verified here.
"""
import argparse
import contextlib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

# The constructor's text encoder is stubbed below; no CLIP operation is used.
sys.modules.setdefault('clip', ModuleType('clip'))
import model.cmdm as cmdm
from utils.model_util import get_model_args


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('expected a plain tensor state dictionary')
    cmdm.Rotation2xyz = lambda **kwargs: torch.nn.Identity()
    cmdm.CMDM.load_and_freeze_clip = lambda self, _: torch.nn.Identity()
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with contextlib.redirect_stdout(sys.stderr):
        model = cmdm.CMDM(**get_model_args(settings, None))
    expected = model.state_dict()
    matched, mismatch = [], []
    for key in sorted(set(state) & set(expected)):
        if tuple(state[key].shape) == tuple(expected[key].shape):
            matched.append(key)
        else:
            mismatch.append({'key': key, 'checkpoint': list(state[key].shape), 'model': list(expected[key].shape)})
    keys = [k for k in state if k.startswith('zero_convs.')]
    max_abs = max((float(state[k].abs().max()) for k in keys if state[k].numel()), default=None)
    report = {'checkpoint': str(args.checkpoint), 'sha256': sha256(args.checkpoint),
              'expected_key_count': len(expected), 'checkpoint_key_count': len(state),
              'matched_key_count': len(matched), 'matched_parameter_elements': sum(state[k].numel() for k in matched),
              'missing_keys': sorted(set(expected)-set(state)), 'unexpected_keys': sorted(set(state)-set(expected)),
              'shape_mismatches': mismatch, 'checkpoint_groups': dict(Counter(k.split('.')[0] for k in state)),
              'control_zero_convs_max_abs': max_abs,
              'caveat': 'Only key/shape compatibility; CLIP and SMPL were stubbed. Not a training or inference result.'}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
