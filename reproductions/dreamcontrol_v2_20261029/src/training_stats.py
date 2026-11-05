"""Streaming normalization statistics for pre-split motion features.

This computes population statistics across valid, unpadded training frames.
It does not perform splitting or validate upstream 263D feature semantics.
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np


def training_stats(clips: Iterable[tuple[str, np.ndarray]], *, feature_dim: int = 263,
                   min_std: float = 1e-6) -> dict:
    """Accumulate only train arrays; each array must contain unpadded frames."""
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError('feature_dim must be a positive integer')
    if not np.isfinite(min_std) or min_std <= 0:
        raise ValueError('min_std must be finite and positive')
    count = 0
    clips_used = 0
    mean = np.zeros(feature_dim, dtype=np.float64)
    m2 = np.zeros(feature_dim, dtype=np.float64)
    for split, values in clips:
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'unknown split: {split!r}')
        if split != 'train':
            continue
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != feature_dim or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError(f'train clip must be nonempty finite (T,{feature_dim})')
        batch_count = len(x)
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        new_count = count + batch_count
        mean += delta * (batch_count / new_count)
        m2 += batch_m2 + delta**2 * (count * batch_count / new_count)
        count = new_count
        clips_used += 1
    if count == 0:
        raise ValueError('no training frames provided')
    raw_std = np.sqrt(np.maximum(m2 / count, 0.0))
    return {'mean': mean, 'std': np.maximum(raw_std, min_std), 'raw_std': raw_std,
            'frame_count': count, 'clip_count': clips_used, 'min_std': min_std,
            'clamped_channels': np.flatnonzero(raw_std < min_std).tolist()}
