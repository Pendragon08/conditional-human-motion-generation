"""Experimental HumanML-format 263D encoder for G1 keypoint trajectories.

Requires the pinned DreamControl TrajGen tree on PYTHONPATH. Uses its HumanML
feature extractor without its human-skeleton rescaling step. Floor inference,
foot threshold and robot point semantics must be validated for each dataset;
these choices are not claimed to be the authors' unpublished v2 implementation.
"""
from __future__ import annotations

import numpy as np

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
          [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
FACE = [2, 1, 17, 16]


def canonicalize_z_up(points: np.ndarray) -> np.ndarray:
    """Convert world Z-up to right-handed Y-up, floor and align initial facing."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (22, 3) or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("expected finite (T,22,3) points with at least two frames")
    p = points[..., [0, 2, 1]].copy()
    p[..., 2] *= -1  # right-handed: [x, z, -y]
    p[..., 1] -= p[..., 1].min()
    p[..., 0] -= p[0, 0, 0]
    p[..., 2] -= p[0, 0, 2]
    across = (p[0, FACE[0]] - p[0, FACE[1]]) + (p[0, FACE[2]] - p[0, FACE[3]])
    across[1] = 0
    if np.linalg.norm(across) < 1e-8:
        raise ValueError("initial hips/shoulders cannot define a facing direction")
    forward = np.cross([0., 1., 0.], across)
    forward /= np.linalg.norm(forward)
    angle = -np.arctan2(forward[0], forward[2])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return p @ rot.T


def encode(points_z_up_world: np.ndarray, *, foot_velocity_squared_threshold: float) -> np.ndarray:
    if not np.isfinite(foot_velocity_squared_threshold) or foot_velocity_squared_threshold <= 0:
        raise ValueError("foot threshold must be positive and finite")
    # This imports pinned DreamControl code, rather than silently substituting
    # a differently normalized third-party HumanML implementation.
    from data_loaders.humanml.scripts.motion_process import extract_features
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
    import torch
    positions = canonicalize_z_up(points_z_up_world)
    # The upstream extractor uses the removed np.float alias. Its restoration
    # is restricted to this synchronous call, then the numpy state is restored.
    had_float = hasattr(np, "float")
    old_float = getattr(np, "float", None)
    if not had_float:
        np.float = float
    try:
        data = extract_features(positions, foot_velocity_squared_threshold,
                                torch.as_tensor(t2m_raw_offsets, dtype=torch.float32),
                                CHAINS, FACE, [8, 11], [7, 10])
    finally:
        if not had_float:
            delattr(np, "float")
        elif old_float is not None:
            np.float = old_float
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (len(positions) - 1, 263) or not np.isfinite(data).all():
        raise ValueError("upstream extractor returned invalid 263D features")
    return data
