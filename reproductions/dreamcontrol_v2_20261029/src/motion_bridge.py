"""Convert near-locked 29-DoF G1 clips to 27-DoF inputs for FK.

This is not the paper's retargeting optimizer or 263D motion encoder. It
rejects clips with active waist roll/pitch instead of silently dropping them.
The output uses Z-up world coordinates, not canonical HumanML coordinates.
"""
from __future__ import annotations

import numpy as np
from g1_fk import URDFFK

SOURCE_NAMES = tuple(name + "_joint" for name in (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw",
))


def prepare_near_locked_clip(motion: np.ndarray, model: URDFFK, *, source_fps: int,
                             target_fps: int, max_locked_angle_rad: float) -> dict:
    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < 2 or not np.isfinite(motion).all():
        raise ValueError("expected a finite (T,36) motion with at least two frames")
    if source_fps <= 0 or target_fps <= 0 or source_fps % target_fps:
        raise ValueError("only exact integer-rate decimation is currently supported")
    if not np.isfinite(max_locked_angle_rad) or max_locked_angle_rad < 0:
        raise ValueError("locked-waist tolerance must be finite and non-negative")
    source_index = {name: index for index, name in enumerate(SOURCE_NAMES)}
    if set(model.actuated) != set(SOURCE_NAMES) - {"waist_roll_joint", "waist_pitch_joint"}:
        raise ValueError("URDF actuated joints do not match the expected G1 27-DoF mapping")
    locked_indices = [7 + source_index[n] for n in ("waist_roll_joint", "waist_pitch_joint")]
    locked_max = float(np.max(np.abs(motion[:, locked_indices])))
    if locked_max > max_locked_angle_rad:
        raise ValueError(f"locked waist deviation {locked_max:.6g} rad exceeds tolerance; re-optimize this clip")
    sample = motion[::source_fps // target_fps]
    if len(sample) < 2:
        raise ValueError("clip is too short after resampling")
    q = sample[:, [7 + source_index[name] for name in model.actuated]]
    quat = sample[:, 3:7]
    norm = np.linalg.norm(quat, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("zero root quaternion")
    quat = quat / norm[:, None]
    points = model.forward(sample[:, :3], quat, q, model.actuated)
    return {"fps": target_fps, "root_pos": sample[:, :3].copy(),
            "root_quat_xyzw": quat, "joint_names": tuple(model.actuated),
            "joint_pos": q, "points_z_up_world": points,
            "max_discarded_waist_angle_rad": locked_max}


def prepare_fitted_clip(motion_at_20hz: np.ndarray, fitted_joint_pos: np.ndarray,
                        fitted_joint_names: tuple[str, ...], model: URDFFK) -> dict:
    """Build FK inputs from explicitly optimized joints and the same source frames.

    The fit's quality must be checked separately; this function verifies only
    shape and identity of the joints and does not prove physical plausibility.
    """
    motion = np.asarray(motion_at_20hz, dtype=float)
    q = np.asarray(fitted_joint_pos, dtype=float)
    names = tuple(fitted_joint_names)
    if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < 2 or not np.isfinite(motion).all():
        raise ValueError("expected finite decimated (T,36) motion")
    if q.shape != (len(motion), 27) or not np.isfinite(q).all():
        raise ValueError("fitted joint angles must be finite and have shape (T,27)")
    if len(set(names)) != 27 or set(names) != set(model.actuated):
        raise ValueError("fitted joint names must match the 27-DoF URDF")
    norm = np.linalg.norm(motion[:, 3:7], axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("zero root quaternion")
    quat = motion[:, 3:7] / norm[:, None]
    points = model.forward(motion[:, :3], quat, q, list(names))
    return {"fps": 20, "root_pos": motion[:, :3].copy(), "root_quat_xyzw": quat,
            "joint_names": names, "joint_pos": q.copy(), "points_z_up_world": points}


def prepare_root_fitted_clip(motion_at_20hz: np.ndarray, fit: dict, model: URDFFK) -> dict:
    """Use *both* the fitted root pose and joints when evaluating root-aware IK.

    This validates field shapes only; fit quality, contacts and dynamics remain
    the caller's responsibility. The original motion and fit are not mutated.
    """
    motion = np.asarray(motion_at_20hz, dtype=float)
    if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < 2 or not np.isfinite(motion).all():
        raise ValueError('expected finite (T,36) source motion')
    root_pos = np.asarray(fit['root_pos'], dtype=float)
    root_quat = np.asarray(fit['root_quat_xyzw'], dtype=float)
    if root_pos.shape != (len(motion), 3) or root_quat.shape != (len(motion), 4):
        raise ValueError('fitted root pose shape does not match motion')
    if not np.isfinite(root_pos).all() or not np.isfinite(root_quat).all():
        raise ValueError('fitted root pose must be finite')
    norms = np.linalg.norm(root_quat, axis=1)
    if not np.allclose(norms, 1., atol=1e-4, rtol=0):
        raise ValueError('fitted root quaternion must be normalized')
    corrected = motion.copy()
    corrected[:, :3], corrected[:, 3:7] = root_pos, root_quat
    return prepare_fitted_clip(corrected, fit['joint_pos'], tuple(fit['joint_names']), model)
