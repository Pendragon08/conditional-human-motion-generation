"""Experimental antialiased resampling of OmniRetarget publisher qpos.

Linear channels use SciPy polyphase FIR. Quaternion signs are made continuous,
their components are FIR-filtered, then normalized. Component filtering is an
approximation on SO(3), not an author-verified method or physical validation.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly


def _align_quaternion_signs(q: np.ndarray) -> np.ndarray:
    dots = np.sum(q[1:] * q[:-1], axis=1)
    flips = np.where(dots < 0, -1., 1.)
    signs = np.concatenate(([1.], np.cumprod(flips)))
    return q * signs[:, None]


def resample_antialias(qpos: np.ndarray, source_fps: float, target_fps: float = 20.,
                       max_seconds: float | None = None) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] not in (34, 36, 41, 43) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('expected finite root plus 27 or 29 joints, optionally with object pose')
    if not all(np.isfinite(x) and x > 0 for x in (source_fps, target_fps)):
        raise ValueError('rates must be finite and positive')
    if target_fps > source_fps:
        raise ValueError('this path only supports equal-rate or downsampled motion')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('maximum duration must be positive and finite')
    ratio = (Fraction(str(target_fps)) / Fraction(str(source_fps))).limit_denominator(1000)
    if abs(float(ratio) - target_fps/source_fps) > 1e-9 or ratio.numerator > 1000:
        raise ValueError('sampling rate ratio is not supported precisely')
    aligned = q.copy()
    object_quaternion = {41: slice(34, 38), 43: slice(36, 40)}.get(q.shape[1])
    quat_slices = [slice(0, 4)] + ([object_quaternion] if object_quaternion else [])
    for s in quat_slices:
        norms = np.linalg.norm(q[:, s], axis=1)
        if not np.allclose(norms, 1., atol=1e-4, rtol=0):
            raise ValueError('source quaternion must be normalized before filtering')
        aligned[:, s] = _align_quaternion_signs(q[:, s])
    # Detrend/re-add the exact endpoint line: phase-dependent FIR gain can
    # otherwise introduce a small drift even for perfectly linear motion.
    source_fraction = np.arange(len(q), dtype=np.float64)[:, None] / (len(q)-1)
    difference = aligned[-1] - aligned[0]
    residual = aligned - (aligned[0] + source_fraction * difference)
    result = resample_poly(residual, ratio.numerator, ratio.denominator, axis=0,
                           window=('kaiser', 5.0), padtype='line')
    target_fraction = (np.arange(len(result), dtype=np.float64) * source_fps /
                       (target_fps * (len(q)-1)))[:, None]
    result += aligned[0] + target_fraction * difference
    available = int(np.floor((len(q)-1)*target_fps/source_fps + 1e-9)) + 1
    if max_seconds is not None:
        available = min(available, int(np.floor(max_seconds*target_fps + 1e-9)))
    result = result[:available].copy()
    if len(result) < 2:
        raise ValueError('motion is too short for target sampling')
    # The first output sample is exactly the first input time. Polyphase FIR
    # edge handling can shift it by centimeters on non-linear trajectories;
    # pin coincident boundaries and keep the impact explicit for evaluation.
    result[0] = aligned[0]
    last_source_index = (len(result)-1) * source_fps / target_fps
    if abs(last_source_index - (len(q)-1)) < 1e-9:
        result[-1] = aligned[-1]
    for s in quat_slices:
        norms = np.linalg.norm(result[:, s], axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError('filtered quaternion has near-zero norm')
        result[:, s] /= norms
    if not np.isfinite(result).all():
        raise ValueError('nonfinite filtered result')
    return result
