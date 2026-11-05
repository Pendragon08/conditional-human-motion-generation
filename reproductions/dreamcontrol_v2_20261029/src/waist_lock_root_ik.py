"""Experimental bounded root-and-joint fitting for 29-to-27 DoF G1 motions.

This fits robot-to-robot keypoints, not the paper's human-to-robot optimizer.
The root correction is part of the output, so callers must not reuse the old
root pose. Contact physics, collision, antialiasing and acceptance are separate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from motion_bridge import SOURCE_NAMES
from waist_lock_ik import _chain, _joint_limits, _positions


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(-1, 3, 3)


def _exp_rotation(v: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(v, dim=-1)
    skew = _skew(v)
    a = torch.sinc(angle / torch.pi)
    b = .5 * torch.sinc(angle / (2 * torch.pi)).square()
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(len(v), -1, -1)
    return eye + a[:, None, None] * skew + b[:, None, None] * (skew @ skew)


def _quat_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[:, :3], a[:, 3:4]
    bv, bw = b[:, :3], b[:, 3:4]
    return torch.cat([aw*bv + bw*av + torch.cross(av, bv, dim=1), aw*bw - (av*bv).sum(1, keepdim=True)], 1)


def _project(v: torch.Tensor, limit: float) -> None:
    length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    v.mul_(torch.clamp(limit / length.clamp_min(1e-12), max=1.0))


def fit_root_and_waist(motion: np.ndarray, urdf29: Path, urdf27: Path, *, steps: int = 100,
                       learning_rate: float = 0.015, nominal_weight: float = 0.002,
                       velocity_weight: float = 0.01, root_weight: float = 0.02,
                       root_velocity_weight: float = 0.01, max_translation_m: float = .08,
                       max_rotation_rad: float = .25, foot_position_weight: float = 0.,
                       contact_velocity_weight: float = 0., contact_speed_mps: float = .03,
                       fps: float = 20., tail_weight: float = 0., tail_fraction: float = .05,
                       return_best: bool = False, ground_penetration_weight: float = 0.,
                       ground_plane_z: float | None = None, ground_contact_weight: float = 0.,
                       ground_contact_band_m: float = .005, device: str = 'cpu') -> dict:
    """Fit 12 source robot keypoints using 27 joints and bounded root deltas."""
    raw = np.asarray(motion, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('expected finite (T,36) motion with at least two frames')
    params = (learning_rate, nominal_weight, velocity_weight, root_weight,
              root_velocity_weight, max_translation_m, max_rotation_rad,
              foot_position_weight, contact_velocity_weight, contact_speed_mps, fps,
              tail_weight, tail_fraction, ground_penetration_weight, ground_contact_weight, ground_contact_band_m)
    if not isinstance(steps, int) or steps < 1 or not all(np.isfinite(params)):
        raise ValueError('invalid optimization configuration')
    if learning_rate <= 0 or fps <= 0 or any(x < 0 for x in params[1:]) or not 0 < tail_fraction <= 1 or ground_contact_band_m <= 0:
        raise ValueError('negative loss weights or bounds, or nonpositive learning rate')
    if ground_plane_z is not None and not np.isfinite(ground_plane_z):
        raise ValueError('ground plane height must be finite')
    if (ground_penetration_weight > 0 or ground_contact_weight > 0) and ground_plane_z is None:
        raise ValueError('a ground plane must be explicitly specified for the experimental penalty')
    norms = np.linalg.norm(raw[:, 3:7], axis=1)
    if np.any(norms < 1e-8) or not np.allclose(norms, 1., atol=1e-4, rtol=0):
        raise ValueError('root quaternions must be normalized')
    torch_device = torch.device(device)
    if torch_device.type not in ('cpu','cuda') or (torch_device.type=='cuda' and torch_device.index is None):
        raise ValueError('explicit CPU or indexed CUDA device required')
    source, target = _chain(urdf29).to(device=torch_device), _chain(urdf27).to(device=torch_device)
    names29, names27 = source.get_joint_parameter_names(), target.get_joint_parameter_names()
    if set(names29) != set(SOURCE_NAMES) or set(names27) != set(SOURCE_NAMES) - {'waist_roll_joint', 'waist_pitch_joint'}:
        raise ValueError('unexpected G1 joint names')
    idx = {name: i for i, name in enumerate(SOURCE_NAMES)}
    q29 = torch.as_tensor(raw[:, [7 + idx[name] for name in names29]],device=torch_device)
    lower, upper = _joint_limits(urdf27, names27)
    lower, upper = lower.to(torch_device), upper.to(torch_device)
    nominal = torch.as_tensor(raw[:, [7 + idx[name] for name in names27]],device=torch_device).clamp(lower, upper)
    if ground_penetration_weight or ground_contact_weight:
        from foot_geometry_audit import foot_spheres
        sphere_offsets, sphere_radii = foot_spheres(urdf27)
        if ground_contact_weight:
            source_offsets, source_radii = foot_spheres(urdf29)
            if not np.allclose(source_offsets,sphere_offsets,rtol=0,atol=1e-12) or not np.allclose(source_radii,sphere_radii,rtol=0,atol=1e-12):
                raise ValueError('source and target foot collision spheres differ')
        sphere_offsets = torch.as_tensor(sphere_offsets,device=torch_device)
        sphere_radii = torch.as_tensor(sphere_radii,device=torch_device)
    with torch.no_grad():
        expected = _positions(source, q29).detach()
        initial = torch.linalg.vector_norm(_positions(target, nominal) - expected, dim=-1)
        original_quat = torch.as_tensor(raw[:, 3:7],device=torch_device)
        original_rotation = _quat_matrix(original_quat)
        original_position = torch.as_tensor(raw[:, :3],device=torch_device)
        expected_world = (original_rotation @ expected.transpose(1, 2)).transpose(1, 2) + original_position[:, None, :]
        reference_foot_delta = expected_world[1:, 4:6] - expected_world[:-1, 4:6]
        contact = torch.linalg.vector_norm(reference_foot_delta, dim=-1) * fps < contact_speed_mps
        if ground_contact_weight:
            src = source.forward_kinematics(q29)
            src_feet = torch.stack([src[name].get_matrix() for name in ('left_ankle_roll_link','right_ankle_roll_link')],dim=1)
            src_centers = src_feet[:,:,:3,3,None] + torch.einsum('tfij,fkj->tfik',src_feet[:,:,:3,:3],sphere_offsets)
            src_centers = src_centers.permute(0,1,3,2)
            src_world = torch.einsum('tij,tfkj->tfki',original_rotation,src_centers) + original_position[:,None,None,:]
            source_clearance = src_world[...,2]-sphere_radii[None]-ground_plane_z
            source_ground_mask = torch.abs(source_clearance) <= ground_contact_band_m
            # Contact target lies on or above the assumed plane, even when the
            # source collision sphere is slightly below it.
            source_contact_target = torch.clamp(source_clearance,min=0.)
    joints = torch.nn.Parameter(nominal.clone())
    delta_position = torch.nn.Parameter(torch.zeros((len(raw), 3), dtype=torch.float64,device=torch_device))
    delta_rotation = torch.nn.Parameter(torch.zeros((len(raw), 3), dtype=torch.float64,device=torch_device))
    optimizer = torch.optim.Adam([joints, delta_position, delta_rotation], lr=learning_rate)

    def objective():
        rotated = _positions(target, joints) @ _exp_rotation(delta_rotation).transpose(1, 2)
        prediction = rotated + delta_position[:, None, :]
        fit_loss = (prediction - expected).square().mean()
        nominal_loss = (joints - nominal).square().mean()
        velocity_loss = ((joints[1:] - joints[:-1]) - (nominal[1:] - nominal[:-1])).square().mean()
        root_loss = delta_position.square().mean() + delta_rotation.square().mean()
        root_vel_loss = (delta_position[1:] - delta_position[:-1]).square().mean() + (delta_rotation[1:] - delta_rotation[:-1]).square().mean()
        foot_loss = (prediction[:, 4:6] - expected[:, 4:6]).square().mean()
        prediction_world = (original_rotation @ prediction.transpose(1, 2)).transpose(1, 2) + original_position[:, None, :]
        predicted_foot_delta = prediction_world[1:, 4:6] - prediction_world[:-1, 4:6]
        difference = predicted_foot_delta - reference_foot_delta
        contact_loss = (difference.square() * contact[:, :, None]).sum() / (contact.sum().clamp_min(1) * 3)
        if tail_weight:
            per_point_loss = (prediction - expected).square().mean(dim=-1).reshape(-1)
            tail_count = max(1, int(np.ceil(len(per_point_loss) * tail_fraction)))
            tail_loss = torch.topk(per_point_loss, tail_count).values.mean()
        else:
            tail_loss = 0.
        if ground_penetration_weight or ground_contact_weight:
            transforms = target.forward_kinematics(joints)
            feet = torch.stack([transforms[name].get_matrix() for name in ('left_ankle_roll_link','right_ankle_roll_link')],dim=1)
            local_centers = feet[:,:,:3,3,None] + torch.einsum('tfij,fkj->tfik',feet[:,:,:3,:3],sphere_offsets)
            local_centers = local_centers.permute(0,1,3,2)
            corrected = torch.einsum('tij,tfkj->tfki',_exp_rotation(delta_rotation),local_centers) + delta_position[:,None,None,:]
            world_centers = torch.einsum('tij,tfkj->tfki',original_rotation,corrected) + original_position[:,None,None,:]
            penetration = torch.clamp(ground_plane_z + sphere_radii[None] - world_centers[...,2],min=0.)
            ground_loss = penetration.square().mean()
            if ground_contact_weight:
                target_clearance = world_centers[...,2]-sphere_radii[None]-ground_plane_z
                contact_height_loss = ((target_clearance-source_contact_target).square()*source_ground_mask).sum() / source_ground_mask.sum().clamp_min(1)
            else:
                contact_height_loss = 0.
        else:
            ground_loss = 0.
            contact_height_loss = 0.
        return (fit_loss + nominal_weight*nominal_loss + velocity_weight*velocity_loss + root_weight*root_loss
                + root_velocity_weight*root_vel_loss + foot_position_weight*foot_loss
                + contact_velocity_weight*contact_loss + tail_weight*tail_loss
                + ground_penetration_weight*ground_loss + ground_contact_weight*contact_height_loss)

    best_loss = float('inf')
    best = None
    best_step = None
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError('non-finite loss')
        if return_best and float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best = tuple(x.detach().clone() for x in (joints, delta_position, delta_rotation))
            best_step = iteration
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            joints.clamp_(lower, upper)
            _project(delta_position, max_translation_m)
            _project(delta_rotation, max_rotation_rad)
    with torch.no_grad():
        if return_best:
            final_loss = float(objective())
            if not np.isfinite(final_loss):
                raise ValueError('non-finite final loss')
            if final_loss < best_loss:
                best_loss, best_step = final_loss, steps
            else:
                for parameter, value in zip((joints, delta_position, delta_rotation), best):
                    parameter.copy_(value)
        err = torch.linalg.vector_norm(_positions(target, joints) @ _exp_rotation(delta_rotation).transpose(1, 2) + delta_position[:, None, :] - expected, dim=-1)
        world_position = torch.as_tensor(raw[:, :3],device=torch_device) + (original_rotation @ delta_position.unsqueeze(-1)).squeeze(-1)
        angle = torch.linalg.vector_norm(delta_rotation, dim=-1)
        delta_quat = torch.cat([.5 * torch.sinc(angle[:, None]/(2*torch.pi)) * delta_rotation,
                                torch.cos(angle[:, None]/2)], dim=1)
        world_quat = _quat_multiply(original_quat, delta_quat)
        return {'joint_names': tuple(names27), 'joint_pos': joints.detach().cpu().numpy().copy(),
                'root_pos': world_position.cpu().numpy().copy(), 'root_quat_xyzw': world_quat.cpu().numpy().copy(),
                'initial_mean_keypoint_error_m': float(initial.mean()),
                'initial_max_keypoint_error_m': float(initial.max()),
                'final_mean_keypoint_error_m': float(err.mean()),
                'final_max_keypoint_error_m': float(err.max()),
                'max_root_translation_m': float(torch.linalg.vector_norm(delta_position, dim=1).max()),
                'max_root_rotation_rad': float(angle.max()),
                'best_objective': best_loss if return_best else None,
                'best_step': best_step}
