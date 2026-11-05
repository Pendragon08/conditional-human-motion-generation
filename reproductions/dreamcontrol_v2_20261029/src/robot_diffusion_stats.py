"""Explicit robot normalization for an existing OmniControl diffusion object.

This does not construct a dataset, train a model, or fix the upstream implicit
hint-mask convention. Robot training requires independently derived train-only
statistics and a verified encoding / spatial hint pipeline.
"""
from __future__ import annotations
from types import MethodType
import numpy as np
import torch


def _scale_for_variable_length(self, mask_hint: torch.Tensor) -> torch.Tensor:
    """Retain upstream 20/max-keyframes rule without the 196-frame assertion."""
    if mask_hint.ndim != 4 or mask_hint.shape[2:] != (22, 1):
        raise ValueError('expected hint mask (batch, time, 22, 1)')
    counts = mask_hint.to(dtype=torch.float32).sum(dim=1).squeeze(-1)
    maximum = counts.max(dim=1).values
    scale = torch.where(maximum > 0, 20. / maximum.clamp_min(1.), torch.zeros_like(maximum))
    return scale[:, None, None, None]


def configure_robot_stats(diffusion, feature_mean, feature_std, spatial_mean, spatial_std) -> None:
    """Set validated 263D and 22x3 train-only stats on one diffusion instance."""
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1)
              for a in (feature_mean, feature_std, spatial_mean, spatial_std)]
    if [len(a) for a in arrays] != [263, 263, 66, 66]:
        raise ValueError('expected feature (263,) and spatial (22,3) statistics')
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('normalization statistics must be finite')
    if np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
        raise ValueError('standard deviations must be positive')
    # Build all values before mutating the object so validation failures cannot
    # leave a partially configured diffusion instance.
    tensors = [torch.as_tensor(a.copy(), dtype=torch.float32) for a in arrays]
    if any(not torch.isfinite(x).all() for x in tensors) or torch.any(tensors[1] <= 0) or torch.any(tensors[3] <= 0):
        raise ValueError('statistics cannot be represented as positive finite float32')
    diffusion.mean, diffusion.std, diffusion.raw_mean, diffusion.raw_std = tensors
    diffusion.calc_grad_scale = MethodType(_scale_for_variable_length, diffusion)
