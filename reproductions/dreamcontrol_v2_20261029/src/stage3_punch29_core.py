"""CPU-testable Punch preparation. Not an Isaac task or a trained policy.

Use the v2 Table IX wrist-origin squared-distance predicate, not the legacy
v1 hand-tip distance threshold. No reference motion is an actor input.
"""
import torch

from stage3_pick29_core import actor_observation

PUNCH_PROMPT = 'a person punches with the right wrist'
TARGET_LOW = (.45, -.20, .95)
TARGET_HIGH = (.55, .10, 1.10)
TARGET_FRAME_LOW = 90
TARGET_FRAME_HIGH = 120
REFERENCE_FPS = 20
WRIST_ERROR_SQUARED_THRESHOLD = .05


def sample_punch_conditions(count, generator=None):
    """Project sampling choice: inclusive integer frames; z-up robot coordinates.

    Conversion into the diffusion model's canonical coordinates remains a
    separate, unvalidated step. This function must not imply it is complete.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('positive integer sample count required')
    low = torch.tensor(TARGET_LOW, dtype=torch.float64)
    high = torch.tensor(TARGET_HIGH, dtype=torch.float64)
    targets = low + torch.rand((count, 3), generator=generator, dtype=torch.float64) * (high-low)
    frames = torch.randint(TARGET_FRAME_LOW, TARGET_FRAME_HIGH+1, (count,), generator=generator)
    return {'right_wrist_target_robot_z_up': targets, 'target_frame': frames,
            'target_seconds': frames.to(torch.float64)/REFERENCE_FPS}


def punch_goal_error_squared(wrist_position, target_position):
    """Both arrays must be N x 3 in the same frame and measured in metres."""
    if wrist_position.ndim != 2 or wrist_position.shape[-1] != 3 or target_position.shape != wrist_position.shape:
        raise ValueError('matching N x 3 positions required; broadcasting is not allowed')
    if not torch.isfinite(wrist_position).all() or not torch.isfinite(target_position).all():
        raise ValueError('positions must be finite')
    return (wrist_position-target_position).square().sum(-1)


def punch_goal_reward(wrist_position, target_position):
    """Table IX term only. This does not prove a complete punching gesture."""
    return (punch_goal_error_squared(wrist_position, target_position)
            <= WRIST_ERROR_SQUARED_THRESHOLD).to(wrist_position.dtype)


def punch_live_observation(angular_velocity, gravity, joint_pos_rel, joint_vel_rel,
                           previous_action, root_pos, root_quat, target_pos, target_quat):
    """29 body joints + 2 hands, live target pose; no reference/time/ID input."""
    return actor_observation(angular_velocity, gravity, joint_pos_rel, joint_vel_rel,
                             previous_action, root_pos, root_quat, target_pos, target_quat)
