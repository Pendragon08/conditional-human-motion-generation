"""Reference-free actor observation prototypes for point and object tasks.

All inputs are live robot state or a task goal. This is not yet wired into
Isaac Lab, and task-specific sensors beyond object pose are not included.
The root quaternion convention is Isaac Lab's wxyz.
"""
from __future__ import annotations

import torch


def actor_point_observation(*, root_pos_w: torch.Tensor, root_quat_wxyz: torch.Tensor,
                            root_ang_vel_b: torch.Tensor, gravity_b: torch.Tensor,
                            joint_pos_rel: torch.Tensor, joint_vel_rel: torch.Tensor,
                            previous_action: torch.Tensor, goal_pos_w: torch.Tensor) -> torch.Tensor:
    tensors = (root_pos_w, root_quat_wxyz, root_ang_vel_b, gravity_b,
               joint_pos_rel, joint_vel_rel, previous_action, goal_pos_w)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("every observation input must be batched (B,D)")
    batch = root_pos_w.shape[0]
    if any(t.shape[0] != batch or t.device != root_pos_w.device or t.dtype != root_pos_w.dtype for t in tensors):
        raise ValueError("observation input batch, device and dtype must match")
    if (root_pos_w.shape[1], root_quat_wxyz.shape[1], root_ang_vel_b.shape[1],
        gravity_b.shape[1], goal_pos_w.shape[1]) != (3, 4, 3, 3, 3):
        raise ValueError("invalid root, orientation, gravity or goal dimensions")
    if joint_pos_rel.shape[1] != 27 or joint_vel_rel.shape[1] != 27 or previous_action.shape[1] != 29:
        raise ValueError("expected 27 body joints and 29 previous action values")
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("observation contains non-finite values")
    norm = torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True)
    if torch.any(norm < 1e-12):
        raise ValueError("zero root quaternion")
    q = root_quat_wxyz / norm
    # Rotate world displacement by inverse unit quaternion into root frame.
    v = goal_pos_w - root_pos_w
    w, xyz = q[:, :1], -q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=1)
    goal_b = v + w * t + torch.cross(xyz, t, dim=1)
    return torch.cat((root_ang_vel_b, gravity_b, joint_pos_rel, joint_vel_rel,
                      previous_action, goal_b), dim=1)


def actor_object_observation(*, root_pos_w: torch.Tensor, root_quat_wxyz: torch.Tensor,
                             root_ang_vel_b: torch.Tensor, gravity_b: torch.Tensor,
                             joint_pos_rel: torch.Tensor, joint_vel_rel: torch.Tensor,
                             previous_action: torch.Tensor, object_pos_w: torch.Tensor,
                             object_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Append live object orientation in root frame; accepts no reference motion.

    The first nonzero quaternion component is canonicalized to be positive.
    Object pose must come from a live task sensor or simulator, never a motion
    reference. This prototype is not wired to an Isaac Lab environment.
    """
    body = actor_point_observation(root_pos_w=root_pos_w, root_quat_wxyz=root_quat_wxyz,
                                   root_ang_vel_b=root_ang_vel_b, gravity_b=gravity_b,
                                   joint_pos_rel=joint_pos_rel, joint_vel_rel=joint_vel_rel,
                                   previous_action=previous_action, goal_pos_w=object_pos_w)
    if (object_quat_wxyz.ndim != 2 or object_quat_wxyz.shape != root_quat_wxyz.shape
            or object_quat_wxyz.device != root_quat_wxyz.device or object_quat_wxyz.dtype != root_quat_wxyz.dtype
            or not torch.isfinite(object_quat_wxyz).all()):
        raise ValueError('object orientation must match finite batched root quaternion')
    root = root_quat_wxyz / torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True)
    object_norm = torch.linalg.vector_norm(object_quat_wxyz, dim=1, keepdim=True)
    if torch.any(object_norm < 1e-12):
        raise ValueError('zero object quaternion')
    obj = object_quat_wxyz / object_norm
    # Hamilton product inverse(root) * object, WXYZ convention.
    rw, rv = root[:, :1], -root[:, 1:]
    ow, ov = obj[:, :1], obj[:, 1:]
    rel_w = rw * ow - (rv * ov).sum(dim=1, keepdim=True)
    rel_v = rw * ov + ow * rv + torch.cross(rv, ov, dim=1)
    rel = torch.cat((rel_w, rel_v), dim=1)
    # Resolve q/-q consistently, including exact 180-degree rotations where
    # the scalar component is zero and the vector component breaks the tie.
    negative = torch.zeros(rel.shape[0], dtype=torch.bool, device=rel.device)
    unresolved = torch.ones_like(negative)
    for i in range(4):
        negative |= unresolved & (rel[:, i] < 0)
        unresolved &= rel[:, i] == 0
    rel = torch.where(negative[:, None], -rel, rel)
    return torch.cat((body, rel), dim=1)
