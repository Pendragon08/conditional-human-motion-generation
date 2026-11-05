"""CPU witness: can padded frame contents affect valid CMDM outputs?

Uses real pretrained model and CLIP with a stubbed SMPL module. Inputs are
synthetic. This is a model behavior check, not robot training or fidelity proof.
"""
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
    torch.set_num_threads(1)
    torch.manual_seed(41)
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8, cond_mode='both_text_spatial',
                               cond_mask_prob=.1, arch='trans_enc', emb_trans_dec=False)
    with contextlib.redirect_stdout(sys.stderr):
        cmdm.Rotation2xyz = StubRotation2xyz
        model = cmdm.CMDM(**get_model_args(settings, None))
        state = torch.load(Path('save/omnicontrol_ckpt/model_humanml3d.pt'), map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('checkpoint mismatch')
        attach_explicit_hint_mask(model)
        model.eval()
        original = torch.randn(1, 263, 1, 12)
        changed = original.clone()
        changed[..., 8:] = torch.randn_like(changed[..., 8:]) * 3
        mask = torch.tensor([True]*8+[False]*4).reshape(1,1,1,12)
        cond = {'mask':mask, 'text':['a person walks forward'], 'hint':torch.zeros(1,12,66),
                'hint_mask':torch.zeros(1,12,22,dtype=torch.bool)}
        with torch.no_grad():
            a = model(original, torch.tensor([500]), y=cond)
            b = model(changed, torch.tensor([500]), y=cond)
        diff = (a[...,:8]-b[...,:8]).abs()
        if not torch.isfinite(diff).all() or diff.max().item() <= 1e-6:
            raise ValueError('expected an observable effect from changing padded input')
        result = {'valid_frames':8,'padded_frames':4,'max_valid_output_difference':float(diff.max()),
                  'mean_valid_output_difference':float(diff.mean()),
                  'warning':'Synthetic padded inputs; real pretrained CMDM and CLIP, SMPL stubbed. No training or physical validation.'}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
