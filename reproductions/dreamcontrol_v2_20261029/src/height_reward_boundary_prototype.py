"""Isolated height reward boundary experiment, not wired to Isaac or training."""
import math
import torch


def height_reward(height, closed, *, height_threshold, fall_threshold):
    """Continuous ramp with caller-provided, explicitly chosen height frame.

    The returned success condition preserves the old strict greater-than rule.
    `height` must already be expressed in the chosen frame; this function does
    not infer a ground plane, actor pose, grasp, or environment origin.
    """
    if not isinstance(height,torch.Tensor) or not torch.is_floating_point(height) or height.ndim!=1 or not torch.isfinite(height).all():
        raise ValueError('height must be a finite floating tensor of shape (N,)')
    if not isinstance(closed,torch.Tensor) or closed.dtype!=torch.bool or closed.shape!=height.shape or closed.device!=height.device:
        raise ValueError('closed must be a boolean tensor with the same shape and device')
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (height_threshold,fall_threshold)) or not fall_threshold<height_threshold:
        raise ValueError('thresholds must be finite and strictly increasing')
    ramp=torch.clamp((height-fall_threshold)/(height_threshold-fall_threshold),min=0.,max=1.)
    return torch.where(closed,ramp,torch.zeros_like(ramp)),closed & (height>height_threshold)
