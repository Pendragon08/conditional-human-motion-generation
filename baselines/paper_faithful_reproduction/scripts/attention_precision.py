"""Opt-in FP64 SDPA inside official DiT attention, with FP32 outputs.

This is a reproduction compatibility experiment, not an author-provided change.
No logit clipping, Q/K normalization, architecture or loss change is introduced.
The original official source files remain unchanged. Each DDP rank installs once.
"""
import contextvars
import functools
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def install():
    from reactivebfm.model.motion_planner.architectures.dit import _DiTAttention
    if getattr(_DiTAttention, '_reproduction_fp64_sdpa', False):
        raise RuntimeError('FP64 SDPA compatibility layer is already installed')
    active = contextvars.ContextVar('reactivebfm_dit_attention', default=False)
    original_attention = _DiTAttention.forward
    original_sdpa = F.scaled_dot_product_attention

    def precise_sdpa(query, key, value, *args, **kwargs):
        if not active.get():
            return original_sdpa(query, key, value, *args, **kwargs)
        if args:
            raise ValueError('Pinned official DiT uses keyword SDPA arguments')
        if query.dtype != torch.float32 or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError('This compatibility variant requires FP32 DiT projections')
        if kwargs.get('dropout_p', 0.0) != 0.0:
            raise ValueError('This witness covers the pinned dropout=0 configuration only')
        promoted_kwargs = dict(kwargs)
        mask = promoted_kwargs.get('attn_mask')
        if mask is not None and mask.is_floating_point():
            promoted_kwargs['attn_mask'] = mask.double()
        with sdpa_kernel(SDPBackend.MATH):
            output = original_sdpa(query.double(), key.double(), value.double(), **promoted_kwargs)
        return output.to(query.dtype)

    @functools.wraps(original_attention)
    def precise_forward(self, *args, **kwargs):
        token = active.set(True)
        try:
            return original_attention(self, *args, **kwargs)
        finally:
            active.reset(token)

    F.scaled_dot_product_attention = precise_sdpa
    _DiTAttention.forward = precise_forward
    _DiTAttention._reproduction_fp64_sdpa = True
    return {'variant': 'DiT SDPA FP64 with FP32 output and parameters',
            'scope': 'official _DiTAttention self and cross attention only',
            'dropout': 0.0, 'architecture_changed': False,
            'loss_changed': False, 'long_run_stability_verified': False}
