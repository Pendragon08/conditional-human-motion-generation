"""Map publisher qpos to the existing G1 fitting input layout.

Drake 1.26.0 inspection of the two publisher URDFs verified their 29 joint
position names match motion_bridge.SOURCE_NAMES. Verify this witness for other
model versions. This rearranges fields only, and does not retarget or resample.
"""
from __future__ import annotations

import numpy as np


def convert_layout(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return robot XYZ+XYZW+29 joints and optional object WXYZ+XYZ pose."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] not in (36, 43) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('expected finite (T,36) or (T,43) with at least 2 frames')
    base_norm = np.linalg.norm(q[:, :4], axis=1)
    if np.any(base_norm < 1e-8):
        raise ValueError('zero base quaternion')
    if not np.allclose(base_norm, 1.0, atol=1e-4, rtol=0):
        raise ValueError('base quaternion must be normalized')
    if q.shape[1] == 43:
        object_norm = np.linalg.norm(q[:, 36:40], axis=1)
        if np.any(object_norm < 1e-8) or not np.allclose(object_norm, 1.0, atol=1e-4, rtol=0):
            raise ValueError('object quaternion must be normalized')
    robot = np.concatenate([q[:, 4:7], q[:, 1:4], q[:, 0:1], q[:, 7:36]], axis=1)
    obj = q[:, 36:43].copy() if q.shape[1] == 43 else None
    return robot, obj
