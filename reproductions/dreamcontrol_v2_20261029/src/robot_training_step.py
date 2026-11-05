"""One experimental robot CMDM optimization step; no job or checkpoint runner.

The explicit-mask MSE guidance is a reproduction approximation, not the
upstream L2 guidance or verified DreamControl-v2 implementation. Caller must
set the trainable parameters, real train-only statistics, model mode, seed,
device, logging and checkpoint policy. Data must be accepted separately.
"""
from __future__ import annotations
import math
import torch
from robot_spatial_guidance import guide


def _motion_loss(model, diffusion, clean, cond, *, guidance_steps: int,
                 guidance_scale: float, timesteps=None, noise=None):
    if clean.ndim != 4 or clean.shape[1:3] != (263, 1) or clean.shape[0] == 0 or clean.shape[-1] < 2:
        raise ValueError('expected a nonempty (B,263,1,T) motion tensor')
    if not torch.is_floating_point(clean) or not torch.isfinite(clean).all():
        raise ValueError('motion must be finite floating point')
    if not isinstance(cond, dict) or not isinstance(cond.get('y'), dict):
        raise ValueError('expected model condition dictionary')
    y = cond['y']
    batch, _, _, frames = clean.shape
    valid, hint, hint_mask = y.get('mask'), y.get('hint'), y.get('hint_mask')
    if not isinstance(valid, torch.Tensor) or valid.shape != (batch, 1, 1, frames) or valid.dtype != torch.bool:
        raise ValueError('expected boolean valid-frame mask (B,1,1,T)')
    if valid.device != clean.device or not valid.all():
        raise ValueError('mixed-length padded batches are unsupported by the unmasked CMDM attention')
    if not isinstance(hint, torch.Tensor) or hint.shape != (batch, frames, 66) or hint.device != clean.device or not torch.isfinite(hint).all():
        raise ValueError('invalid spatial hint')
    if not isinstance(hint_mask, torch.Tensor) or hint_mask.shape != (batch, frames, 22) or hint_mask.device != clean.device or hint_mask.dtype != torch.bool:
        raise ValueError('invalid explicit hint mask')
    if not isinstance(y.get('text'), list) or len(y['text']) != batch or any(not isinstance(s, str) or not s for s in y['text']):
        raise ValueError('one nonempty text string per sample is required')
    if not isinstance(guidance_steps, int) or isinstance(guidance_steps, bool) or guidance_steps < 0 or not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError('invalid guidance configuration')
    if getattr(diffusion.model_mean_type, 'name', None) != 'START_X':
        raise ValueError('this step supports clean-motion prediction only')
    nsteps = len(diffusion.posterior_log_variance_clipped)
    if timesteps is None:
        timesteps = torch.randint(nsteps, (batch,), device=clean.device)
    if (not isinstance(timesteps, torch.Tensor) or timesteps.shape != (batch,) or
            timesteps.dtype not in (torch.int32, torch.int64) or timesteps.device != clean.device or
            torch.any(timesteps < 0) or torch.any(timesteps >= nsteps)):
        raise ValueError('invalid diffusion timesteps')
    if noise is None:
        noise = torch.randn_like(clean)
    if noise.shape != clean.shape or noise.device != clean.device or noise.dtype != clean.dtype or not torch.isfinite(noise).all():
        raise ValueError('invalid diffusion noise')
    noisy = diffusion.q_sample(clean, timesteps, noise=noise)
    if guidance_steps:
        noisy = guide(diffusion, noisy, timesteps, hint, hint_mask,
                      step_size=guidance_scale, steps=guidance_steps, valid_frames=valid[:, 0, 0, :])
    prediction = model(noisy, timesteps, y=y)
    if prediction.shape != clean.shape or not torch.isfinite(prediction).all():
        raise ValueError('invalid model prediction')
    loss = (prediction - clean).square().mean()
    if not torch.isfinite(loss):
        raise ValueError('non-finite loss')
    return loss, {'loss':float(loss.detach()), 'batch_size':batch, 'frames':frames,
                  'timesteps':timesteps.detach().cpu().tolist(), 'guidance_steps':guidance_steps}


def train_accumulated_step(model, diffusion, optimizer, batches, *, guidance_steps: int,
                           guidance_scale: float):
    """One approximate per-sample-mean update from unpadded microbatches.

    Equal samples, rather than equal frames, receive equal weight. This is not
    evidence of the unreleased author's v2 batching policy.
    """
    batches = list(batches)
    if not batches or any(not isinstance(x, (tuple, list)) or len(x) != 2 or
                          not isinstance(x[0], torch.Tensor) or x[0].ndim != 4 or
                          x[0].shape[0] == 0 for x in batches):
        raise ValueError('expected nonempty motion/condition microbatches')
    total = sum(x[0].shape[0] for x in batches)
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    try:
        for clean, cond in batches:
            loss, detail = _motion_loss(model, diffusion, clean, cond,
                                        guidance_steps=guidance_steps, guidance_scale=guidance_scale)
            (loss * (clean.shape[0] / total)).backward()
            if isinstance(cond['y'].get('seq_name'), list):
                detail['sample_ids'] = list(cond['y']['seq_name'])
            metrics.append(detail)
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('missing or non-finite trainable gradients')
    optimizer.step()
    return {'loss':sum(m['loss'] * m['batch_size'] for m in metrics) / total,
            'batch_size':total, 'microbatches':metrics, 'guidance_steps':guidance_steps}


def train_step(model, diffusion, optimizer, clean, cond, *, guidance_steps: int,
               guidance_scale: float, timesteps=None, noise=None):
    optimizer.zero_grad(set_to_none=True)
    loss, detail = _motion_loss(model, diffusion, clean, cond, guidance_steps=guidance_steps,
                                guidance_scale=guidance_scale, timesteps=timesteps, noise=noise)
    loss.backward()
    grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('missing or non-finite trainable gradients')
    optimizer.step()
    return detail
