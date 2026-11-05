"""URDF forward kinematics for paper Table VII G1 keypoints.

Inputs are world XYZ, xyzw quaternions and joint angles by explicit URDF joint
name. Outputs remain world XYZ in the URDF's Z-up convention; HumanML's
canonical Y-up conversion and 263D feature encoding are a separate step.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

PAPER_22_LINKS = (
    "pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link",
    "imu_link", "left_knee_link", "right_knee_link", "torso_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "imu_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "mid360_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "d435_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)


def _rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation, batch first."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("joint rotation axis is zero")
    axis /= norm
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.asarray(angle, dtype=float)
    c = np.cos(angle)[..., None, None]
    s = np.sin(angle)[..., None, None]
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * cross


def _xyz(value: str) -> np.ndarray:
    v = np.fromstring(value, sep=" ", dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"invalid URDF vector: {value}")
    return v


def _rpy(value: str) -> np.ndarray:
    r, p, y = _xyz(value)
    return _rotation([0, 0, 1], y) @ _rotation([0, 1, 0], p) @ _rotation([1, 0, 0], r)


def _quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("root quaternion must have shape (T,4) and finite values")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError("zero root quaternion")
    x, y, z, w = (q / norm).T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(-1, 3, 3)


class URDFFK:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.links = {node.attrib["name"] for node in root.findall("link")}
        self.joints = []
        child_links = set()
        for node in root.findall("joint"):
            kind = node.attrib["type"]
            if kind not in {"fixed", "revolute", "continuous"}:
                raise ValueError(f"unsupported joint type {kind}")
            parent = node.find("parent").attrib["link"]
            child = node.find("child").attrib["link"]
            origin = node.find("origin")
            axis_node = node.find("axis")
            xyz = _xyz(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            rot = _rpy(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
            axis = _xyz(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0")
            self.joints.append((node.attrib["name"], kind, parent, child, xyz, rot, axis))
            child_links.add(child)
        roots = self.links - child_links
        if len(roots) != 1:
            raise ValueError(f"expected one robot root, got {sorted(roots)}")
        self.root = roots.pop()
        self.actuated = [j[0] for j in self.joints if j[1] != "fixed"]

    def forward(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray,
                joint_angles: np.ndarray, joint_names: list[str],
                links=PAPER_22_LINKS, *, return_rotations: bool = False):
        pos = np.asarray(root_pos, dtype=float)
        q = np.asarray(joint_angles, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
            raise ValueError("root positions must be finite (T,3)")
        if q.shape != (len(pos), len(joint_names)) or not np.isfinite(q).all():
            raise ValueError("joint angles do not match frame count and names")
        if len(joint_names) != len(set(joint_names)) or set(joint_names) != set(self.actuated):
            raise ValueError("explicit joint names must match all actuated URDF joints exactly")
        if not set(links) <= self.links:
            raise ValueError(f"URDF missing links {sorted(set(links) - self.links)}")
        rotations = {self.root: _quaternion_xyzw(root_quat_xyzw)}
        if len(rotations[self.root]) != len(pos):
            raise ValueError("root position/quaternion frame mismatch")
        positions = {self.root: pos}
        index = {name: i for i, name in enumerate(joint_names)}
        pending = self.joints.copy()
        while pending:
            remaining = []
            for name, kind, parent, child, offset, origin_rot, axis in pending:
                if parent not in rotations:
                    remaining.append((name, kind, parent, child, offset, origin_rot, axis))
                    continue
                parent_r = rotations[parent]
                positions[child] = positions[parent] + np.einsum("tij,j->ti", parent_r, offset)
                joint_r = parent_r @ origin_rot
                rotations[child] = joint_r if kind == "fixed" else joint_r @ _rotation(axis, q[:, index[name]])
            if len(remaining) == len(pending):
                raise ValueError("URDF is disconnected or cyclic")
            pending = remaining
        result = np.stack([positions[name] for name in links], axis=1)
        if return_rotations:
            return result, np.stack([rotations[name] for name in links], axis=1)
        return result
