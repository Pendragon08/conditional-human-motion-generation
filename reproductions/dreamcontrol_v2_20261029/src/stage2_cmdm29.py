"""Pinned OmniControl CMDM, MDM initialization and explicit padding support.

The unused SMPL mesh helper is removed from an in-memory copy of the source:
263D motion and spatial decoding use recover_from_ric, never SMPL. No model
parameters are replaced by stubs. The original checkout remains unchanged.
"""
import hashlib
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def padding_forward(self, x, timesteps, y=None):
    if y is None or x.ndim != 4 or x.shape[1:3] != (263, 1):
        raise ValueError('263D motion and conditions required')
    b, _, _, n = x.shape
    valid = y['mask'][:, 0, 0]
    if valid.shape != (b, n) or valid.dtype != torch.bool or not valid.any(-1).all():
        raise ValueError('invalid valid-frame mask')
    key_mask = torch.cat((torch.zeros((b, 1), device=x.device, dtype=torch.bool), ~valid), dim=1)
    text = y.get('text_embedding')
    if text is None:
        text = self.encode_text(y['text'])
    text = self.mask_cond(text, force_mask=y.get('uncond', False))
    mask = y['hint_mask'] & valid.unsqueeze(-1)
    hint = y['hint'].reshape(b, n, 22, 3) * mask.unsqueeze(-1)
    hint = hint.reshape(b, n, 66)
    control_emb = self.c_embed_timestep(timesteps) + self.c_embed_text(text)
    control_x = self.c_input_process(x)
    control_x = control_x + self.input_hint_block(hint.float()) * mask.any(-1).T.unsqueeze(-1)
    control_x = self.c_sequence_pos_encoder(torch.cat((control_emb, control_x), dim=0))
    layers = self.c_seqTransEncoder(control_x, src_key_padding_mask=key_mask)
    control = torch.stack([module(layers[i]) for i, module in enumerate(self.zero_convs)])
    base_emb = self.embed_timestep(timesteps) + self.embed_text(text)
    inputs = self.sequence_pos_encoder(torch.cat((base_emb, self.input_process(x)), dim=0))
    output = self.seqTransEncoder(inputs, src_key_padding_mask=key_mask, control=control)[1:]
    return self.output_process(output) * valid[:, None, None]


def build_model(cfg):
    original = Path(cfg['trajgen']) / 'model/cmdm.py'
    if sha(original) != cfg['cmdm_source_sha256']:
        raise ValueError('pinned CMDM source changed')
    source = original.read_text()
    edits = {
        'from model.rotation2xyz import Rotation2xyz': '# 263D path does not construct the unused SMPL mesh helper',
        "self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)": 'self.rot2xyz = None',
        '        self.rot2xyz.smpl_model._apply(fn)': '        # No mesh helper is used by the 263D network.',
        '        self.rot2xyz.smpl_model.train(*args, **kwargs)': '        # Frozen CLIP mode is handled by the caller.',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError('unexpected CMDM helper source: ' + old)
        source = source.replace(old, new)
    module = ModuleType('stage2_cmdm_vendor')
    module.__package__ = 'model'
    module.__file__ = str(original)
    sys.modules[module.__name__] = module
    exec(compile(source, str(original) + '[263D-no-unused-mesh]', 'exec'), module.__dict__)
    from utils.model_util import get_model_args
    settings = SimpleNamespace(dataset='humanml', latent_dim=512, layers=8,
                               cond_mode='both_text_spatial', cond_mask_prob=.1,
                               arch='trans_enc', emb_trans_dec=False)
    model = module.CMDM(**get_model_args(settings, None))
    weight_path = Path(cfg['initial_checkpoint'])
    if sha(weight_path) != cfg['initial_checkpoint_sha256']:
        raise ValueError('MDM initialization bytes changed')
    weights = torch.load(weight_path, map_location='cpu', weights_only=True)
    missing, extra = model.load_state_dict(weights, strict=False)
    allowed = ('clip_model.', 'input_hint_block.', 'zero_convs.')
    if extra or any(not k.startswith(allowed) for k in missing):
        raise ValueError('MDM/CMDM initialization key mismatch')
    # The official OmniControl MDM release includes a copied c_* backbone.
    copied = 0
    for key, value in weights.items():
        if key.startswith('c_'):
            base_key = key[2:]
            if base_key not in weights or not torch.equal(value, weights[base_key]):
                raise ValueError('control backbone is not a copy of MDM: ' + key)
            copied += 1
    if not copied:
        raise ValueError('expected official MDM control-backbone copy')
    if any(torch.count_nonzero(p) for p in model.zero_convs.parameters()):
        raise ValueError('control output blocks must start at zero')
    if any(torch.count_nonzero(p) for p in model.input_hint_block.poseEmbedding[-1].parameters()):
        raise ValueError('hint output must start at zero')
    model.forward = MethodType(padding_forward, model)
    model.clip_model.eval()
    receipt = {'source_sha256': sha(original), 'executed_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'mdm_sha256': sha(weight_path), 'mdm_keys': len(weights),
               'copied_control_keys': copied, 'fresh_zero_output_layers': len(model.zero_convs),
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'clip_frozen': all(not p.requires_grad for p in model.clip_model.parameters()),
               'smpl_helper_used': False, 'architecture': 'OmniControl CMDM 8x512, explicit hint mask and attention padding',
               'scope': '29DoF dataset adaptation, not author-v2 released training code'}
    return model, receipt


def compact_state(model):
    """Frozen CLIP is referenced by its pinned asset hash, not copied per save."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('clip_model.')}


def restore_state(model, state):
    if set(state) != {k for k in model.state_dict() if not k.startswith('clip_model.')}:
        raise ValueError('checkpoint network keys mismatch')
    missing, extra = model.load_state_dict(state, strict=False)
    if extra or any(not k.startswith('clip_model.') for k in missing):
        raise ValueError('checkpoint network mismatch')
