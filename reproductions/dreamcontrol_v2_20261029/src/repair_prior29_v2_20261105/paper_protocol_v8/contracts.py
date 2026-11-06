"""Shared task-mask and frame contracts for the repaired robot prior."""
import hashlib
import json

import numpy as np


TASK_POINTS = {
    "pick": ((0, 19, 21), (4, 5)),
    "pour": ((0, 21), (10, 11)),
    "bimanual_box_lift": ((0, 20, 21), (10, 11)),
    "box_push_pull": ((0, 20, 21), (10, 11)),
}
GUIDANCE_POLICY = {
    "loss": "sum_of_masked_point_euclidean_distances",
    "train": {"steps": 20, "below_20_steps": 100, "scale": 0.5},
    "sample": {"steps": 10, "below_10_steps": 500, "density_scale": 20.0},
    "posterior_variance_floor": 0.01,
    "freeze_feature_frame_zero": True,
    "upstream_commit": "ec28b508313d6837de52c68247fbc8cf81fe4935",
}
COORDINATE_SCHEMA = "training_floor_canonical_v2"


def task_mask(task, frames):
    """Construct the same dense/sparse layout for training and generation."""
    dense, sparse = TASK_POINTS[task]
    mask = np.zeros((frames, 22), dtype=bool)
    mask[:, dense] = True
    mask[::5, sparse] = True
    return mask


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def coordinate_contract(initial_world, joint_order, urdf_sha256, ground_z=0.0):
    """Match feature263.canonicalize_z_up for a stationary scene template."""
    initial = np.asarray(initial_world, dtype=np.float64)
    if initial.shape != (22, 3) or not np.isfinite(initial).all():
        raise ValueError("expected 22 finite world-frame standing points")
    z_up_to_y_up = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    points = initial @ z_up_to_y_up.T
    across = (points[2] - points[1]) + (points[17] - points[16])
    across[1] = 0.0
    if np.linalg.norm(across) < 1e-8:
        raise ValueError("standing template cannot define a facing direction")
    forward = np.cross([0., 1., 0.], across)
    angle = -np.arctan2(forward[0], forward[2])
    c, s = np.cos(angle), np.sin(angle)
    heading = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    world_to_canonical = heading @ z_up_to_y_up
    origin = np.array([initial[0, 0], initial[0, 1], initial[:, 2].min()])
    forward_translation = -origin @ world_to_canonical.T
    info = {
        "schema": COORDINATE_SCHEMA,
        "ground_z": float(ground_z),
        "height_origin_world_m": float(origin[2]),
        "initial_pelvis_world_m": initial[0].tolist(),
        "standing_keypoints_sha256": hashlib.sha256(initial.astype("<f8").tobytes()).hexdigest(),
        "joint_order_sha256": json_digest(list(joint_order)),
        "urdf_sha256": urdf_sha256,
    }
    transform = {
        "canonical_to_world_rotation": world_to_canonical.T.tolist(),
        "canonical_to_world_translation": origin.tolist(),
        "world_to_canonical_rotation": world_to_canonical.tolist(),
        "world_to_canonical_translation": forward_translation.tolist(),
    }
    info["calibration_sha256"] = json_digest({**info, **transform})
    return {**transform, "coordinate_provenance": info}


def world_to_canonical(points, transform):
    return (np.asarray(points) @ np.asarray(transform["world_to_canonical_rotation"]).T
            + np.asarray(transform["world_to_canonical_translation"]))


def canonical_to_world(points, transform):
    return (np.asarray(points) @ np.asarray(transform["canonical_to_world_rotation"]).T
            + np.asarray(transform["canonical_to_world_translation"]))

