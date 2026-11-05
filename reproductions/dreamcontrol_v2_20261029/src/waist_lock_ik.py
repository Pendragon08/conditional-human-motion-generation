"""Experimental 29-to-27 DoF G1 keypoint fitting (not paper retargeting).

Uses the old 29-DoF robot's keypoints as targets and fits a locked-waist robot
with differentiable PyTorch kinematics. Hyperparameters are explicit because
the v2 paper does not disclose the optimizer's complete specification. This
does not solve human-to-G1 retargeting or independently guarantee dynamics.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import pytorch_kinematics as pk

from motion_bridge import SOURCE_NAMES

FIT_LINKS = (
    'left_hip_pitch_link', 'right_hip_pitch_link',
    'left_knee_link', 'right_knee_link',
    'left_ankle_roll_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'right_shoulder_roll_link',
    'left_elbow_link', 'right_elbow_link',
    'left_wrist_yaw_link', 'right_wrist_yaw_link',
)


def _chain(path: Path):
    root = ET.parse(path).getroot()
    # pytorch_kinematics does not parse MuJoCo-only URDF extensions.
    for node in root.findall('mujoco'):
        root.remove(node)
    return pk.build_chain_from_urdf(ET.tostring(root, encoding='unicode')).to(dtype=torch.float64, device='cpu')


def _joint_limits(path: Path, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    tree = ET.parse(path).getroot()
    nodes = {node.get('name'): node for node in tree.findall('joint')}
    limits = []
    for name in names:
        joint = nodes[name]
        limit = joint.find('limit')
        if joint.get('type') not in ('revolute', 'continuous'):
            raise ValueError(f'unsupported actuator {name}')
        if joint.get('type') == 'continuous':
            limits.append((-np.pi, np.pi))
        elif limit is None or limit.get('lower') is None or limit.get('upper') is None:
            raise ValueError(f'missing limits for {name}')
        else:
            limits.append((float(limit.get('lower')), float(limit.get('upper'))))
    lower, upper = torch.tensor(limits, dtype=torch.float64).T
    if torch.any(lower >= upper):
        raise ValueError('invalid URDF joint limits')
    return lower, upper


def _positions(chain, angles: torch.Tensor) -> torch.Tensor:
    transforms = chain.forward_kinematics(angles)
    return torch.stack([transforms[name].get_matrix()[:, :3, 3] for name in FIT_LINKS], dim=1)


def fit_locked_waist(motion: np.ndarray, urdf29: Path, urdf27: Path, *, steps: int,
                     learning_rate: float, nominal_weight: float, velocity_weight: float) -> dict:
    raw = np.asarray(motion, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('expected finite (T,36) source motion')
    if steps < 1 or learning_rate <= 0 or nominal_weight < 0 or velocity_weight < 0:
        raise ValueError('invalid optimization hyperparameters')
    source = _chain(urdf29)
    target = _chain(urdf27)
    names29, names27 = source.get_joint_parameter_names(), target.get_joint_parameter_names()
    if set(names29) != set(SOURCE_NAMES) or set(names27) != set(SOURCE_NAMES) - {'waist_roll_joint', 'waist_pitch_joint'}:
        raise ValueError('joint names do not match the expected G1 29/27 mapping')
    src_idx = {name: i for i, name in enumerate(SOURCE_NAMES)}
    q29 = torch.as_tensor(raw[:, [7 + src_idx[name] for name in names29]])
    lower, upper = _joint_limits(urdf27, names27)
    nominal = torch.as_tensor(raw[:, [7 + src_idx[name] for name in names27]]).clamp(lower, upper)
    with torch.no_grad():
        expected = _positions(source, q29).detach()
        initial_error = torch.linalg.vector_norm(_positions(target, nominal) - expected, dim=-1).mean().item()
    q = torch.nn.Parameter(nominal.clone())
    optimizer = torch.optim.Adam([q], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        fit_loss = (_positions(target, q) - expected).square().mean()
        nominal_loss = (q - nominal).square().mean()
        velocity_loss = ((q[1:] - q[:-1]) - (nominal[1:] - nominal[:-1])).square().mean()
        loss = fit_loss + nominal_weight * nominal_loss + velocity_weight * velocity_loss
        if not torch.isfinite(loss):
            raise ValueError('non-finite optimization loss')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q.clamp_(lower, upper)
    with torch.no_grad():
        errors = torch.linalg.vector_norm(_positions(target, q) - expected, dim=-1)
        result = q.detach().cpu().numpy().copy()
    return {'joint_names': tuple(names27), 'joint_pos': result,
            'initial_mean_keypoint_error_m': initial_error,
            'final_mean_keypoint_error_m': float(errors.mean()),
            'final_max_keypoint_error_m': float(errors.max())}
