"""Experimental bridge from canonical 22-point diffusion output to old PyRoki.

This is not author DreamControl-v2 post-optimization. The caller supplies the
old solver and every scene-dependent quantity, including the coordinate
transform and ground; nothing here establishes that those choices are true.
"""
from __future__ import annotations

import numpy as np


def postopt_canonical_motion(points, *, solver, foot_detector, robot, robot_collision,
                             heightmap, human_indices, robot_indices, connection_mask,
                             weights, canonical_to_solver_rotation, canonical_to_solver_translation,
                             pick_point_solver, foot_displacement_sq_threshold,
                             joint_names, joint_lower, joint_upper, fps):
    """Call the old v1 solver on one generated trajectory, dropping frame 0.

    ``foot_detector`` has the original signature and returns per-foot values
    for T-1 displacements. A positive value is used as a contact *proxy* as in
    the old script; it does not establish physical contact. The selected
    ground and the caller's rigid transform must be independently validated.
    """
    p = np.asarray(points)
    if p.ndim != 3 or p.shape[1:] != (22, 3) or p.shape[0] < 2 or not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all():
        raise ValueError('expected finite floating (T,22,3) points with T >= 2')
    R = np.asarray(canonical_to_solver_rotation, dtype=np.float64)
    t = np.asarray(canonical_to_solver_translation, dtype=np.float64)
    pick = np.asarray(pick_point_solver, dtype=np.float64)
    if (R.shape != (3, 3) or t.shape != (3,) or pick.shape != (3,) or
        not all(np.isfinite(x).all() for x in (R, t, pick)) or
        not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6) or
        not np.isclose(np.linalg.det(R), 1, rtol=0, atol=1e-6)):
        raise ValueError('an explicit proper rigid transform and pick point are required')
    if (isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0 or
        isinstance(foot_displacement_sq_threshold, bool) or
        not np.isfinite(foot_displacement_sq_threshold) or foot_displacement_sq_threshold < 0):
        raise ValueError('invalid explicit FPS or squared foot displacement threshold')
    names = tuple(joint_names)
    lower, upper = np.asarray(joint_lower, dtype=float), np.asarray(joint_upper, dtype=float)
    if (len(names) != 27 or len(set(names)) != 27 or any(not isinstance(n, str) or not n for n in names) or
        lower.shape != (27,) or upper.shape != (27,) or not np.isfinite(lower).all() or
        not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError('27 unique joint names and finite ordered limits are required')
    actuated_names = getattr(getattr(robot, 'joints', None), 'actuated_names', None)
    if actuated_names is None or tuple(actuated_names) != names:
        raise ValueError('joint names must match the solver robot actuator order')
    if not callable(solver) or not callable(foot_detector) or not callable(getattr(heightmap, 'project_points', None)):
        raise ValueError('explicit original-solver compatible callbacks are required')
    # Match the old Pick_sim script's T->T-1 alignment while making the axis
    # transform an explicit input rather than assuming it is world calibration.
    transformed = (p.astype(np.float64) @ R.T + t).astype(np.float32)
    left, right = foot_detector(transformed[None], thres=float(foot_displacement_sq_threshold))
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (1, len(p)-1) or right.shape != left.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('foot detector returned invalid interval values')
    target = transformed[1:]
    left_points = heightmap.project_points(target[:, 10, :])
    right_points = heightmap.project_points(target[:, 11, :])
    if any(np.asarray(x).shape != (len(p)-1, 3) or not np.isfinite(np.asarray(x)).all()
           for x in (left_points, right_points)):
        raise ValueError('ground projection returned invalid points')
    root, joints = solver(robot=robot, robot_coll=robot_collision, target_keypoints=target,
        is_left_foot_contact=left[0] > 0, is_right_foot_contact=right[0] > 0,
        left_foot_keypoints=left_points, right_foot_keypoints=right_points,
        smpl_joint_retarget_indices=human_indices, g1_joint_retarget_indices=robot_indices,
        smpl_mask=connection_mask, heightmap=heightmap, weights=weights, pick_point=pick.astype(np.float32))
    q = np.asarray(joints)
    rq = np.asarray(root.wxyz_xyz)
    if (q.shape != (len(p)-1, 27) or rq.shape != (len(p)-1, 7) or
        not np.isfinite(q).all() or not np.isfinite(rq).all() or
        not np.allclose(np.linalg.norm(rq[:, :4], axis=1), 1, rtol=0, atol=1e-4)):
        raise ValueError('solver returned invalid root pose or joint state')
    max_violation = float(max(0., np.max(lower-q), np.max(q-upper)))
    return {'root_wxyz_xyz': rq.copy(), 'joint_pos': q.copy(), 'joint_names': names,
            'input_frame_indices': np.arange(1, len(p), dtype=np.int64),
            'left_foot_low_speed_proxy': (left[0] > 0), 'right_foot_low_speed_proxy': (right[0] > 0),
            'metadata': {'fps': float(fps), 'max_joint_limit_violation_rad': max_violation,
                         'foot_displacement_sq_threshold': float(foot_displacement_sq_threshold),
                         'method': 'original-v1 solver adapter; approximate and not author v2',
                         'coordinate_transform': R.tolist(), 'translation': t.tolist(),
                         'training_or_physical_acceptance': False}}
