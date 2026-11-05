"""Explicit-mask MSE spatial guidance for experimental 263D robot motions.

Uses the project's HumanML position decoder, not a full G1 joint-space FK.
This function alone does not change the model's separate realism branch, which
still uses an implicit hint mask and requires its own integration and audit.
"""
from __future__ import annotations
import numpy as np
import torch


def guide(diffusion, x: torch.Tensor, timesteps: torch.Tensor, normalized_hint: torch.Tensor,
          hint_mask: torch.Tensor, *, step_size: float, steps: int,
          valid_frames: torch.Tensor | None = None, min_variance: float = .01) -> torch.Tensor:
    """Apply bounded-count per-sample gradients against explicitly masked goals.

    x is Bx263x1xT, hint is BxTx66, mask is BxTx22 (or BxTx22x1).
    An explicit mask can represent a valid goal whose normalized XYZ is zero.
    Hints are denormalized with the configured train-only spatial statistics.
    """
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    if x.ndim != 4 or x.shape[1:3] != (263, 1) or not torch.isfinite(x).all():
        raise ValueError('expected finite x with shape (B,263,1,T)')
    batch, _, _, frames = x.shape
    if normalized_hint.shape != (batch, frames, 66) or not torch.isfinite(normalized_hint).all():
        raise ValueError('hint must be finite and have shape (B,T,66)')
    if hint_mask.shape == (batch, frames, 22):
        hint_mask = hint_mask.unsqueeze(-1)
    if hint_mask.shape != (batch, frames, 22, 1) or hint_mask.dtype != torch.bool:
        raise ValueError('hint_mask must be boolean with shape (B,T,22[,1])')
    if valid_frames is not None:
        if valid_frames.shape != (batch, frames) or valid_frames.dtype != torch.bool:
            raise ValueError('valid_frames must be boolean with shape (B,T)')
        hint_mask = hint_mask & valid_frames.to(device=hint_mask.device)[:, :, None, None]
    if not isinstance(steps, int) or steps < 0 or not np.isfinite(step_size) or step_size < 0 or not np.isfinite(min_variance) or min_variance < 0:
        raise ValueError('invalid guidance settings')
    if timesteps.shape != (batch,) or timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError('timesteps must be integer (B,)')
    if torch.any(timesteps < 0) or torch.any(timesteps >= len(diffusion.posterior_log_variance_clipped)):
        raise ValueError('timestep outside diffusion schedule')
    device, dtype = x.device, x.dtype
    mean, std = diffusion.mean.to(device=device, dtype=dtype), diffusion.std.to(device=device, dtype=dtype)
    spatial_mean = diffusion.raw_mean.to(device=device, dtype=dtype)
    spatial_std = diffusion.raw_std.to(device=device, dtype=dtype)
    if mean.numel() != 263 or std.numel() != 263 or spatial_mean.numel() != 66 or spatial_std.numel() != 66:
        raise ValueError('robot diffusion statistics are not configured')
    if any(not torch.isfinite(v).all() for v in (mean, std, spatial_mean, spatial_std)) or torch.any(std <= 0) or torch.any(spatial_std <= 0):
        raise ValueError('normalization statistics invalid')
    mask = hint_mask.to(device=device)
    target = (normalized_hint.to(device=device, dtype=dtype) * spatial_std + spatial_mean).reshape(batch, frames, 22, 3)
    count = mask.sum(dim=(1, 2, 3)).clamp_min(1).to(dtype=dtype) * 3
    log_variance = torch.as_tensor(diffusion.posterior_log_variance_clipped, dtype=dtype, device=device)
    variance = log_variance[timesteps.to(device=device, dtype=torch.long)].exp().clamp_min(min_variance).reshape(batch, 1, 1, 1)
    guided = x.detach().clone()
    for _ in range(steps):
        with torch.enable_grad():
            work = guided.detach().requires_grad_(True)
            features = work.permute(0, 3, 2, 1).squeeze(2) * std + mean
            points = recover_from_ric(features, 22)
            residual = (points - target) * mask
            loss = (residual.square().sum(dim=(1, 2, 3)) / count).sum()
            gradient = torch.autograd.grad(loss, work)[0]
            # Preserve upstream convention that initial frame is the origin.
            gradient[..., 0] = 0
        guided = guided - step_size * variance * gradient
    return guided.detach()
