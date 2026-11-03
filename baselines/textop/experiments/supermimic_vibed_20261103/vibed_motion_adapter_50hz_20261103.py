"""Convert validated TextOp motion arrays to the vibed CSV column contract.

Coordinate-system and robot-asset equivalence must be established separately.
This module does not read motion files or launch training.
"""

import numpy as np
from supermimic_textop_order_contract_v1 import reorder_indices

G1_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]


def motion_to_csv_matrix(arrays, source_joint_names, source_body_names):
    fps = np.asarray(arrays['fps'])
    if fps.size != 1 or float(fps.reshape(-1)[0]) != 50.0:
        raise ValueError('source must be sampled at 50 Hz')
    if len(set(source_body_names)) != len(source_body_names) or source_body_names.count('pelvis') != 1:
        raise ValueError('expected exactly one pelvis body')
    order = reorder_indices(source_joint_names, G1_JOINT_NAMES, 29)
    joints = np.asarray(arrays['joint_pos'])
    positions = np.asarray(arrays['body_pos_w'])
    quats = np.asarray(arrays['body_quat_w'])
    if joints.ndim != 2 or joints.shape[1] != 29 or joints.shape[0] < 2:
        raise ValueError('expected at least two frames of 29 joints')
    n, bodies = joints.shape[0], len(source_body_names)
    if positions.shape != (n, bodies, 3) or quats.shape != (n, bodies, 4):
        raise ValueError('body array shape mismatch')
    if any(x.dtype.kind != 'f' or not np.isfinite(x).all() for x in (joints, positions, quats)):
        raise ValueError('motion values must be finite floating point')
    root = source_body_names.index('pelvis')
    q_wxyz = quats[:, root, :]
    if not np.all(np.abs(np.linalg.norm(q_wxyz, axis=1) - 1.0) <= 1e-3):
        raise ValueError('invalid pelvis quaternion')
    # vibed CSV expects position, xyzw quaternion, then joint positions.
    return np.concatenate((positions[:, root, :], q_wxyz[:, (1, 2, 3, 0)], joints[:, order]), axis=1)
