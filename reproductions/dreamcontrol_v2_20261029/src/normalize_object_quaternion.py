"""Explicit, auditable object quaternion normalization for OmniRetarget.

Positive scaling of a nonzero quaternion leaves its represented orientation
unchanged after normalization. This does not validate the object pose, dynamics,
or whether the upstream non-unit values are the intended orientation.
"""
from __future__ import annotations
import numpy as np


def normalize_object_quaternions(qpos: np.ndarray, *, min_norm: float = .5,
                                 max_norm: float = 2., tolerance: float = 1e-4) -> tuple[np.ndarray, dict]:
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 43 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('expected finite (T,43) qpos with an object')
    if not all(np.isfinite(x) for x in (min_norm, max_norm, tolerance)) or not (0 < min_norm <= 1 <= max_norm) or tolerance < 0:
        raise ValueError('invalid normalization policy')
    base_norm = np.linalg.norm(q[:, :4], axis=1)
    if not np.allclose(base_norm, 1., atol=tolerance, rtol=0):
        raise ValueError('root quaternion invalid; this policy only repairs object quaternions')
    norm = np.linalg.norm(q[:, 36:40], axis=1)
    if np.any(norm < min_norm) or np.any(norm > max_norm):
        raise ValueError('object quaternion norm outside explicit safety bounds')
    # Normalize all frames if any fail tolerance, so downstream values stay
    # continuous and exact in the presence of tiny numerical fluctuations.
    changed = np.abs(norm - 1.) > tolerance
    result = q.copy()
    if np.any(changed):
        result[:, 36:40] = q[:, 36:40] / norm[:, None]
    return result, {'policy': 'normalize_nonzero_object_quaternion',
                    'min_allowed_norm': min_norm, 'max_allowed_norm': max_norm,
                    'tolerance': tolerance, 'frames_outside_tolerance': int(changed.sum()),
                    'frames_normalized': len(q) if np.any(changed) else 0,
                    'original_norm_min': float(norm.min()), 'original_norm_max': float(norm.max()),
                    'max_abs_component_change': float(np.max(np.abs(result[:, 36:40] - q[:, 36:40]))),
                    'warning': 'Orientation intent and physical plausibility remain unverified.'}
