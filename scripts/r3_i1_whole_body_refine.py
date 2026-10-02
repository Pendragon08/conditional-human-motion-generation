"""Jointly refine a retargeted G1 motion against a fixed spatial condition.

This is a *new* whole-body/contact/temporal pilot. It does not call or reuse
the retired right-leg-only IK correction. It never changes human generation,
the original retargeting output, or the pretrained tracking policy.

The source hint is mapped into a robot-reachable, root-relative coordinate
system with a leg-length ratio fixed from the source motion and G1 URDF. Both
raw and refined references are scored against this same precomputed target.
The unscaled world-space hint error is separately reported for transparency.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jaxlie
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


ROTATE_TO_G1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
RIGHT_FOOT_SMPL = 11
SMPL_PELVIS = 0
SMPL_RIGHT_HIP = 2
SMPL_RIGHT_KNEE = 5
SMPL_RIGHT_ANKLE = 8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def squared_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        if not torch.any(mask):
            return value.sum() * 0.0
        value = value[mask]
    return value.square().sum(dim=-1).mean()


def smootherstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return value * value * (3 - 2 * value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-joint-change-rad", type=float, default=0.25)
    parser.add_argument("--max-root-xy-change-m", type=float, default=0.06)
    parser.add_argument("--max-root-z-change-m", type=float, default=0.02)
    parser.add_argument("--velocity-weight", type=float, default=3.0)
    parser.add_argument("--acceleration-weight", type=float, default=4.0)
    parser.add_argument("--peak-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1 or args.max_samples < 1:
        raise ValueError("steps and max-samples must be positive")
    stage = args.stage.resolve()
    source = stage / "Kick_sim/results.npy"
    raw_dir = stage / "Kick_sim1"
    retarget_source = stage / "Kick_sim/retarget.py"
    urdf = stage.parents[1] / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
    motion_data = np.load(source, allow_pickle=True).item()
    human = np.asarray(motion_data["motion"]).transpose((0, 3, 1, 2)) @ ROTATE_TO_G1.T
    hints = np.asarray(motion_data["hint"]) @ ROTATE_TO_G1.T
    joint_names = ast.literal_eval(
        re.search(r"joint_names = (\[.*?\])", retarget_source.read_text(), re.S).group(1)
    )
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") and limit.get("upper"):
            joint_limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]), float(limit.attrib["upper"])
            )
    device = torch.device(args.device)
    lower = torch.tensor([joint_limits[name][0] for name in joint_names], device=device)
    upper = torch.tensor([joint_limits[name][1] for name in joint_names], device=device)
    chain = pk.build_chain_from_urdf(urdf.read_text()).to(dtype=torch.float32, device=device)

    def fk(joints: torch.Tensor, root: torch.Tensor, orientation: torch.Tensor):
        cfg = {name: joints[:, j] for j, name in enumerate(joint_names)}
        transforms = chain.forward_kinematics(cfg)
        out = {}
        for name in FOOT_LINKS:
            local = transforms[name].get_matrix()
            world = torch.matmul(orientation, local[:, :3, :3])
            position = torch.einsum("tij,tj->ti", orientation, local[:, :3, 3]) + root
            out[name] = (position, world)
        return out

    # URDF-derived dimensionless mapping, fixed before any candidate is scored.
    zero = {name: torch.zeros(1, device=device) for name in joint_names}
    neutral = chain.forward_kinematics(zero)
    robot_hip = neutral["right_hip_pitch_link"].get_matrix()[0, :3, 3]
    robot_knee = neutral["right_knee_link"].get_matrix()[0, :3, 3]
    robot_ankle = neutral["right_ankle_roll_link"].get_matrix()[0, :3, 3]
    robot_leg_m = float(torch.linalg.norm(robot_hip - robot_knee) + torch.linalg.norm(robot_knee - robot_ankle))

    paths = sorted(raw_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    rows = []
    for path in paths[: args.max_samples]:
        started = time.perf_counter()
        index = int(path.stem)
        with path.open("rb") as stream:
            motion = pickle.load(stream)
        q0 = torch.as_tensor(np.asarray(motion["joints"]), dtype=torch.float32, device=device)
        root0 = torch.as_tensor(np.asarray(motion["global_position"]), dtype=torch.float32, device=device)
        quat = np.asarray(motion["global_pose"].rotation().wxyz)
        orientation = torch.as_tensor(
            Rotation.from_quat(np.roll(quat, -1, axis=1)).as_matrix(),
            dtype=torch.float32, device=device,
        )
        if len(q0) != human.shape[1] - 1 or len(q0) != len(hints[index]) - 1:
            raise ValueError(f"Frame alignment mismatch in {path}")
        source_motion = human[index, 1:]
        human_leg = (
            np.linalg.norm(source_motion[:, SMPL_RIGHT_HIP] - source_motion[:, SMPL_RIGHT_KNEE], axis=-1)
            + np.linalg.norm(source_motion[:, SMPL_RIGHT_KNEE] - source_motion[:, SMPL_RIGHT_ANKLE], axis=-1)
        )
        human_leg_m = float(np.median(human_leg))
        if not 0.4 < human_leg_m < 1.4:
            raise ValueError(f"Unexpected human leg length in {path}: {human_leg_m}")
        scale = robot_leg_m / human_leg_m
        hint = hints[index, 1:, RIGHT_FOOT_SMPL]
        mask_np = np.linalg.norm(hint, axis=-1) > 1e-6
        if not np.any(mask_np):
            raise ValueError(f"Missing pre-generation right-foot hint in {path}")
        mask = torch.as_tensor(mask_np, device=device)
        hint_world = torch.as_tensor(hint, dtype=torch.float32, device=device)
        pelvis_world = torch.as_tensor(source_motion[:, SMPL_PELVIS], dtype=torch.float32, device=device)
        # The Kick hint is a single fixed point repeated over its constrained
        # frames. Fix one mapped scene goal BEFORE refinement; using root0[t]
        # here would incorrectly turn that point into a moving target.
        hinted_positions = hint_world[mask]
        hint_range = hinted_positions.max(dim=0).values - hinted_positions.min(dim=0).values
        if float(hint_range.max()) > 1e-4:
            raise ValueError(f"Expected a fixed Kick target, found moving hint in {path}")
        constrained = torch.where(mask)[0]
        anchor_frame = int(constrained[len(constrained) // 2])
        target_xyz = (root0[anchor_frame].detach()
                      + scale * (hint_world[anchor_frame] - pelvis_world[anchor_frame]))
        target = target_xyz[None].expand_as(root0)

        with torch.no_grad():
            original = fk(q0, root0, orientation)
            left0, left_rot0 = original[FOOT_LINKS[0]]
            right0, right_rot0 = original[FOOT_LINKS[1]]
            raw_error = float(torch.linalg.norm(right0[mask] - target[mask], dim=-1).mean())
            unscaled_error = float(torch.linalg.norm(right0[mask] - hint_world[mask], dim=-1).mean())
            # Contact classification is frozen from the original trajectory.
            # It is only a geometric stance proxy, not a measured force label.
            min_height = torch.minimum(left0[:, 2], right0[:, 2]).quantile(0.1)
            left_speed = torch.linalg.norm(left0[1:] - left0[:-1], dim=-1)
            right_speed = torch.linalg.norm(right0[1:] - right0[:-1], dim=-1)
            left_contact = (left0[:, 2] < min_height + 0.08) & (
                torch.cat((left_speed[:1], left_speed)) < 0.08
            )
            right_contact = (right0[:, 2] < min_height + 0.08) & (
                torch.cat((right_speed[:1], right_speed)) < 0.08
            ) & ~mask

        frames = torch.arange(len(q0), dtype=torch.float32, device=device)
        first, last = int(constrained[0]), int(constrained[-1])
        envelope = smootherstep((frames - (first - 20)) / 20) * smootherstep(((last + 20) - frames) / 20)
        joint_parameter = torch.nn.Parameter(torch.zeros_like(q0))
        root_parameter = torch.nn.Parameter(torch.zeros_like(root0))
        optimizer = torch.optim.Adam([joint_parameter, root_parameter], lr=0.03)

        def candidate():
            dq = args.max_joint_change_rad * envelope[:, None] * torch.tanh(joint_parameter)
            root_cap = torch.tensor(
                [args.max_root_xy_change_m, args.max_root_xy_change_m, args.max_root_z_change_m],
                dtype=torch.float32, device=device,
            )
            dr = root_cap[None] * envelope[:, None] * torch.tanh(root_parameter)
            q = torch.clamp(q0 + dq, lower, upper)
            return q, root0 + dr, dq, dr

        initial_peak_frame = int(torch.argmax((right0 - root0)[:, 0]))
        initial_peak_forward_m = float((right0 - root0)[:, 0].max())
        for _ in range(args.steps):
            optimizer.zero_grad()
            q, root, dq, dr = candidate()
            feet = fk(q, root, orientation)
            left, left_rot = feet[FOOT_LINKS[0]]
            right, right_rot = feet[FOOT_LINKS[1]]
            # One objective, optimized jointly over the whole body and root.
            # Scale each term to similar squared-metre or squared-radian units.
            target_loss = squared_mean(right - target, mask)
            stance_loss = squared_mean(left - left0, left_contact) + squared_mean(right - right0, right_contact)
            stance_orientation_loss = squared_mean(left_rot[:, :, 2] - left_rot0[:, :, 2], left_contact)
            pose_loss = dq.square().mean() + 3.0 * dr.square().mean()
            velocity_loss = (dq[1:] - dq[:-1]).square().mean() + 3.0 * (dr[1:] - dr[:-1]).square().mean()
            acceleration_loss = (dq[2:] - 2 * dq[1:-1] + dq[:-2]).square().mean()
            # Preserve the original kick's non-target phases and event timing.
            outside_event_loss = squared_mean((right - root) - (right0 - root0), ~mask)
            peak_loss = squared_mean((right - root)[initial_peak_frame] - (right0 - root0)[initial_peak_frame])
            floor_loss = torch.relu(min_height - torch.minimum(left[:, 2], right[:, 2])).square().mean()
            loss = (
                1.0 * target_loss + 3.0 * stance_loss + 0.3 * stance_orientation_loss
                + 0.8 * pose_loss + args.velocity_weight * velocity_loss
                + args.acceleration_weight * acceleration_loss
                + 0.4 * outside_event_loss + args.peak_weight * peak_loss + 2.0 * floor_loss
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            q, root, dq, dr = candidate()
            feet = fk(q, root, orientation)
            left, _ = feet[FOOT_LINKS[0]]
            right, _ = feet[FOOT_LINKS[1]]
            after_error = float(torch.linalg.norm(right[mask] - target[mask], dim=-1).mean())
            stance_drift = float(torch.linalg.norm(left[left_contact] - left0[left_contact], dim=-1).mean()) if left_contact.any() else None
            right_stance_drift = float(torch.linalg.norm(right[right_contact] - right0[right_contact], dim=-1).mean()) if right_contact.any() else None
            peak_frame = int(torch.argmax((right - root)[:, 0]))
            peak_forward_m = float((right - root)[:, 0].max())
            joint_speed_delta = float((dq[1:] - dq[:-1]).abs().max())
            joint_accel_delta = float((dq[2:] - 2 * dq[1:-1] + dq[:-2]).abs().max())
            limit_violations = int(((q < lower - 1e-4) | (q > upper + 1e-4)).sum())
        exported = None
        if args.export_dir:
            destination = args.export_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace {destination}")
            revised = dict(motion)
            revised["joints"] = q.cpu().numpy().astype(np.asarray(motion["joints"]).dtype, copy=False)
            revised["global_position"] = root.cpu().numpy().astype(np.asarray(motion["global_position"]).dtype, copy=False)
            # Retain orientation and task metadata; update SE3 translation for
            # consumers that read global_pose rather than global_position.
            revised["global_pose"] = jaxlie.SE3.from_rotation_and_translation(
                motion["global_pose"].rotation(), jnp.asarray(revised["global_position"])
            )
            with destination.open("xb") as stream:
                pickle.dump(revised, stream, protocol=pickle.HIGHEST_PROTOCOL)
            exported = {"name": destination.name, "sha256": digest(destination)}
        row = {
            "sample": index,
            "source_sha256": digest(path),
            "scale_robot_leg_over_human_leg": scale,
            "external_target": "pre-generation fixed right-foot hint, pelvis-relative at hint-median frame, scaled by input/URDF leg lengths, anchored to raw robot root at that frame",
            "mapped_fixed_goal_xyz_m": target_xyz.cpu().tolist(),
            "goal_anchor_frame": anchor_frame,
            "hint_frame_count": int(mask.sum()),
            "raw_mapped_target_error_m": raw_error,
            "refined_mapped_target_error_m": after_error,
            "raw_unscaled_world_hint_error_m": unscaled_error,
            "left_stance_drift_m": stance_drift,
            "right_stance_drift_m": right_stance_drift,
            "left_stance_frames": int(left_contact.sum()),
            "right_stance_frames": int(right_contact.sum()),
            "peak_forward_frame_shift": peak_frame - initial_peak_frame,
            "kick_peak_forward_amplitude_ratio": peak_forward_m / initial_peak_forward_m if abs(initial_peak_forward_m) > 1e-6 else None,
            "max_upper_body_joint_change_rad": float(dq[:, 12:].abs().max()),
            "max_joint_change_rad": float(dq.abs().max()),
            "max_joint_speed_change_rad_per_frame": joint_speed_delta,
            "max_joint_accel_change_rad_per_frame2": joint_accel_delta,
            "max_root_change_m": float(dr.abs().max()),
            "joint_limit_violations": limit_violations,
            "elapsed_wall_s": time.perf_counter() - started,
            "exported_reference": exported,
        }
        rows.append(row)
        print(f"sample {index}: mapped target {raw_error:.3f} -> {after_error:.3f} m", flush=True)

    result = {
        "status": "whole_body_contact_temporal_fixed_scene_target_component",
        "target_is_fixed_per_motion": True,
        "source_npy_sha256": digest(source),
        "source_retarget_script_sha256": digest(retarget_source),
        "urdf_sha256": digest(urdf),
        "steps": args.steps,
        "velocity_weight": args.velocity_weight,
        "acceleration_weight": args.acceleration_weight,
        "peak_weight": args.peak_weight,
        "max_samples": args.max_samples,
        "robot_leg_length_m": robot_leg_m,
        "rows": rows,
        "limitations": [
            "No new diffusion generator or controller is trained by this script",
            "Stance labels are geometric proxies; no force or torque constraints in optimizer",
            "Pelvis-relative morphology mapping must be reported separately from unscaled world hints",
            "A simulator test and matched-budget PPO are required before I1 Gate can pass",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
