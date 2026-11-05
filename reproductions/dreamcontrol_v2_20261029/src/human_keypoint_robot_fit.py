"""Experimental 12-keypoint SMPL-X to G1 refinement initialized by robot IK.

Uses PyTorch CPU kinematics, not the paper's unavailable PyRoki optimizer.
Targets use explicit human limb scales and joint positions, with approximate
bone-direction and foot-velocity losses; they are not the undisclosed paper
loss or validated contact labels. Output is exploratory and never auto-approved.
"""
from pathlib import Path
import numpy as np
import torch
from motion_bridge import SOURCE_NAMES
from waist_lock_ik import _chain, _joint_limits, _positions
from waist_lock_root_ik import _exp_rotation, _quat_matrix, _quat_multiply

HUMAN_KEYPOINT_INDICES = (1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)
SEGMENTS = ((0, 2), (1, 3), (2, 4), (3, 5), (6, 8), (7, 9), (8, 10), (9, 11))


def fit_human_keypoints(human_joints, initial_motion, urdf27: Path, *, fps=20.,
                        leg_scale=.83, arm_scale=.738, pelvis_scale=.83,
                        steps=150, learning_rate=.01, nominal_weight=.01,
                        direction_weight=.01, foot_velocity_weight=1.,
                        temporal_weight=.01, root_rotation_limit=.2,
                        human_contact_speed=.05):
    human = np.asarray(human_joints, dtype=np.float64)
    initial = np.asarray(initial_motion, dtype=np.float64)
    if human.ndim != 3 or human.shape[1] < 22 or human.shape[2] != 3 or len(human) < 2:
        raise ValueError('human joints must be (T,>=22,3)')
    if initial.shape != (len(human), 34) or not np.isfinite(initial).all() or not np.isfinite(human).all():
        raise ValueError('initial motion must be finite xyz+xyzw+27 joints aligned with human')
    values = (fps, leg_scale, arm_scale, pelvis_scale, learning_rate, nominal_weight,
              direction_weight, foot_velocity_weight, temporal_weight, root_rotation_limit, human_contact_speed)
    if not all(np.isfinite(values)) or any(x < 0 for x in values) or min(fps, leg_scale, arm_scale, pelvis_scale, learning_rate) <= 0 or not isinstance(steps, int) or steps < 1:
        raise ValueError('invalid configuration')
    if not np.allclose(np.linalg.norm(initial[:, 3:7], axis=1), 1., atol=1e-4, rtol=0):
        raise ValueError('root quaternions must be unit length')
    chain = _chain(urdf27)
    names = chain.get_joint_parameter_names()
    expected = set(SOURCE_NAMES) - {'waist_roll_joint', 'waist_pitch_joint'}
    if set(names) != expected:
        raise ValueError('unexpected 27-DoF robot joints')
    nominal_order = [n for n in SOURCE_NAMES if n in expected]
    lower, upper = _joint_limits(urdf27, names)
    nominal = torch.as_tensor(initial[:, [7 + nominal_order.index(n) for n in names]]).clamp(lower, upper)
    human = torch.as_tensor(human)
    scales = torch.tensor([leg_scale] * 6 + [arm_scale] * 6, dtype=torch.float64)
    target = (human[:, HUMAN_KEYPOINT_INDICES] - human[:, :1]) * scales[None, :, None]
    human_world_feet = human[:, :1] * pelvis_scale + target[:, 4:6]
    human_delta = human_world_feet[1:] - human_world_feet[:-1]
    contacts = torch.linalg.vector_norm(human_delta, dim=-1) * fps < human_contact_speed
    root = torch.as_tensor(initial[:, :3])
    original_q = torch.as_tensor(initial[:, 3:7])
    rotation = _quat_matrix(original_q)
    q = torch.nn.Parameter(nominal.clone())
    delta = torch.nn.Parameter(torch.zeros((len(human), 3), dtype=torch.float64))
    optim = torch.optim.Adam([q, delta], lr=learning_rate)
    left = torch.tensor([x[0] for x in SEGMENTS]); right = torch.tensor([x[1] for x in SEGMENTS])

    def predict():
        return torch.einsum('tij,tkj->tki', rotation @ _exp_rotation(delta), _positions(chain, q))

    def loss():
        p = predict()
        pos_loss = (p - target).square().mean()
        pd, hd = p[:, right]-p[:, left], target[:, right]-target[:, left]
        pd = pd / torch.linalg.vector_norm(pd, dim=-1, keepdim=True).clamp_min(1e-8)
        hd = hd / torch.linalg.vector_norm(hd, dim=-1, keepdim=True).clamp_min(1e-8)
        direction = (pd-hd).square().mean()
        foot_world = p[:, 4:6] + root[:, None, :]
        feet_delta = foot_world[1:] - foot_world[:-1]
        foot = ((feet_delta-human_delta).square() * contacts[:, :, None]).sum() / (contacts.sum().clamp_min(1) * 3)
        nominal_l = (q-nominal).square().mean() + delta.square().mean()
        temporal = ((q[1:]-q[:-1])-(nominal[1:]-nominal[:-1])).square().mean()
        return pos_loss + direction_weight*direction + foot_velocity_weight*foot + nominal_weight*nominal_l + temporal_weight*temporal

    with torch.no_grad():
        before = torch.linalg.vector_norm(predict()-target, dim=-1).numpy().copy()
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        value = loss()
        if not torch.isfinite(value):
            raise ValueError('non-finite loss')
        value.backward(); optim.step()
        with torch.no_grad():
            q.clamp_(lower, upper)
            size = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            delta.mul_(torch.clamp(root_rotation_limit / size.clamp_min(1e-12), max=1.))
    with torch.no_grad():
        after = torch.linalg.vector_norm(predict()-target, dim=-1).numpy().copy()
        angle = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        dq = torch.cat((delta * (.5*torch.sinc(angle/(2*torch.pi))), torch.cos(angle/2)), dim=1)
        quat = _quat_multiply(original_q, dq).numpy()
    return {'joint_names': tuple(names), 'joint_pos': q.detach().numpy(), 'root_pos': root.numpy(),
            'root_quat_xyzw': quat, 'target_relative_world': target.numpy(),
            'initial_mean_error_m': float(before.mean()), 'final_mean_error_m': float(after.mean()),
            'initial_max_error_m': float(before.max()), 'final_max_error_m': float(after.max()),
            'low_speed_foot_intervals': int(contacts.sum()), 'steps': steps}
