"""Experimental 30-to-20 Hz time interpolation for OmniRetarget qpos.

Joint order is preserved but has NOT been verified against a Drake plant. This
is interpolation only: it does not perform locked-waist optimization or apply
an antialias filter, and is not asserted to match the unpublished v2 pipeline.
"""
from __future__ import annotations

import numpy as np


def _slerp(a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.sum(a * b, axis=1)
    b = np.where((dot < 0)[:, None], -b, b)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    near = sine < 1e-7
    safe_sine = np.where(near, 1.0, sine)
    wa = np.sin((1.0 - alpha) * angle) / safe_sine
    wb = np.sin(alpha * angle) / safe_sine
    wa = np.where(near, 1.0 - alpha, wa)
    wb = np.where(near, alpha, wb)
    out = wa[:, None] * a + wb[:, None] * b
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def resample(qpos: np.ndarray, source_fps: float, target_fps: float = 20.0,
             max_seconds: float | None = 10.0) -> np.ndarray:
    """Sample on a target time grid without extrapolation, up to max_seconds.

    Inputs and output use publisher format: WXYZ+XYZ+29 joints, with an
    optional WXYZ+XYZ object pose. Original input is never modified.
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] not in (36, 43) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('expected finite (T,36) or (T,43) qpos with at least two frames')
    if not all(np.isfinite(v) and v > 0 for v in (source_fps, target_fps)):
        raise ValueError('rates must be positive and finite')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('maximum duration must be positive and finite or None')
    quat_slices = [slice(0, 4)] + ([slice(36, 40)] if q.shape[1] == 43 else [])
    for s in quat_slices:
        norm = np.linalg.norm(q[:, s], axis=1)
        if np.any(norm < 1e-8):
            raise ValueError('zero quaternion in source')
        if not np.allclose(norm, 1.0, atol=1e-4, rtol=0):
            raise ValueError('source quaternion is not normalized; inspect before any correction')
    available = int(np.floor((len(q) - 1) * target_fps / source_fps + 1e-9)) + 1
    if max_seconds is None:
        n = available
    else:
        max_count = int(np.floor(max_seconds * target_fps + 1e-9))
        if max_count < 2:
            raise ValueError('duration must permit at least two target frames')
        n = min(max_count, available)
    if n < 2:
        raise ValueError('motion is too short for the target rate')
    source_index = np.arange(n, dtype=np.float64) * (source_fps / target_fps)
    lo = np.floor(source_index).astype(int)
    hi = np.minimum(lo + 1, len(q) - 1)
    alpha = source_index - lo
    result = q[lo] * (1.0 - alpha[:, None]) + q[hi] * alpha[:, None]
    for s in quat_slices:
        result[:, s] = _slerp(q[lo, s], q[hi, s], alpha)
    return result


def split_resampled(qpos: np.ndarray, source_fps: float, target_fps: float = 20.0,
                    segment_seconds: float = 10.0) -> tuple[list[np.ndarray], int]:
    """Resample once, then split without overlap; explicitly count short tails.

    Group all returned segments by their original source motion during dataset
    partitioning to avoid putting related motion into multiple data splits.
    A one-frame tail cannot represent motion and is counted as dropped.
    """
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError('target rate must be positive and finite')
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError('segment duration must be positive and finite')
    frames_per_segment = int(np.floor(segment_seconds * target_fps + 1e-9))
    if frames_per_segment < 2:
        raise ValueError('segment must permit at least two frames')
    full = resample(qpos, source_fps, target_fps, max_seconds=None)
    segments = []
    dropped = 0
    for start in range(0, len(full), frames_per_segment):
        clip = full[start:start + frames_per_segment]
        if len(clip) < 2:
            dropped += len(clip)
        else:
            segments.append(clip.copy())
    return segments, dropped
