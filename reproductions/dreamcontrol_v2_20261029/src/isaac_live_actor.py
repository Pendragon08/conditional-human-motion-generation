"""Isaac Lab observation-term adapter; reads live state, not motion references.

The adapter may be imported without Isaac Lab for CPU contract tests. A real
ObservationManager must resolve SceneEntityCfg before calling this function.
This is an experimental interface, not a validated v2 policy or checkpoint.
"""
from __future__ import annotations

from numbers import Integral
from actor_observation import actor_object_observation

G1_27_JOINT_NAMES = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
)


def live_object_policy_observation(env, *, robot_cfg, object_cfg):
    """Read one 96-value policy observation from resolved live Isaac assets.

    `robot_cfg.joint_ids` must be resolved to the exact 27-joint order above.
    Quaternion fields use Isaac Lab's WXYZ convention. The action manager's
    `action` field matches Isaac Lab's `mdp.last_action` at observation time.
    No motion reference, reference-derived hand target, time or reward is read.
    Object pose is a simulator state; real-world perception is outside scope.
    """
    if robot_cfg.name == object_cfg.name:
        raise ValueError('robot and object must be different scene assets')
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ids = robot_cfg.joint_ids
    available = robot.joint_names
    if isinstance(ids, slice):
        # SceneEntityCfg deliberately collapses a full, matching joint list to
        # slice(None). It is safe only when both declared and actual full order
        # exactly match our required 27-joint list.
        declared = getattr(robot_cfg, 'joint_names', None)
        if ids != slice(None) or declared is None or tuple(declared) != G1_27_JOINT_NAMES or tuple(available) != G1_27_JOINT_NAMES:
            raise ValueError('joint IDs must be resolved and ordered')
        ids = list(range(27))
    ids = list(ids)
    if len(ids) != 27 or len(set(ids)) != 27:
        raise ValueError('exactly 27 distinct joint IDs are required')
    if any(not isinstance(i, Integral) or i < 0 or i >= len(available) for i in ids):
        raise ValueError('invalid joint ID')
    ids = [int(i) for i in ids]
    if tuple(available[i] for i in ids) != G1_27_JOINT_NAMES:
        raise ValueError('resolved joint order differs from required G1 order')
    data = robot.data
    return actor_object_observation(
        root_pos_w=data.root_pos_w,
        root_quat_wxyz=data.root_quat_w,
        root_ang_vel_b=data.root_ang_vel_b,
        gravity_b=data.projected_gravity_b,
        joint_pos_rel=data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
        joint_vel_rel=data.joint_vel[:, ids] - data.default_joint_vel[:, ids],
        previous_action=env.action_manager.action,
        object_pos_w=obj.data.root_pos_w,
        object_quat_wxyz=obj.data.root_quat_w,
    )
