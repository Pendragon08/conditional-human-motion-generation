"""Small robot-space condition-correction ablation on fixed Kick references.

Optimizes only the right leg of already-retargeted G1 motion. Human generation,
root trajectory, and the support leg remain untouched. This is an I1 component
pilot, not a diffusion-model update, dynamics test, or final I1 validation.
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
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


ROTATE_TO_G1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
RIGHT_FOOT_SMPL = 11
RIGHT_LEG_NAMES = (
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--mode", choices=("dense", "window_smooth"), default="window_smooth")
    parser.add_argument("--max-joint-change-rad", type=float, default=1.0,
                        help="Window mode: positive cap; 0 reproduces uncapped exploratory run")
    parser.add_argument("--export-dir", type=Path,
                        help="Optional new directory for corrected G1 references; never overwrites an existing file")
    args = parser.parse_args()
    if args.max_samples < 1 or args.steps < 1:
        raise ValueError("Positive max-samples and steps required")
    stage = args.stage.resolve()
    export_dir = args.export_dir.resolve() if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
    source = (stage / "Kick_sim/results.npy").resolve()
    data = np.load(source, allow_pickle=True).item()
    hints = np.asarray(data["hint"])
    retarget_source = stage / "Kick_sim/retarget.py"
    joint_names = ast.literal_eval(
        re.search(r"joint_names = (\[.*?\])", retarget_source.read_text(), re.S).group(1)
    )
    leg_indices = [joint_names.index(name) for name in RIGHT_LEG_NAMES]
    urdf = stage.parents[1] / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    limits = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") and limit.get("upper"):
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    lower = torch.tensor([limits[name][0] for name in RIGHT_LEG_NAMES])
    upper = torch.tensor([limits[name][1] for name in RIGHT_LEG_NAMES])
    chain = pk.build_chain_from_urdf(urdf.read_text())
    paths = sorted((stage / "Kick_sim1").glob("*.pkl"), key=lambda p: int(p.stem))
    rows = []
    for path in paths[: args.max_samples]:
        started = time.perf_counter()
        index = int(path.stem)
        with path.open("rb") as stream:
            robot = pickle.load(stream)
        q0 = torch.as_tensor(np.asarray(robot["joints"]), dtype=torch.float32)
        root = torch.as_tensor(np.asarray(robot["global_position"]), dtype=torch.float32)
        quat = np.asarray(robot["global_pose"].rotation().wxyz)
        rotation = torch.as_tensor(Rotation.from_quat(np.roll(quat, -1, axis=1)).as_matrix(), dtype=torch.float32)
        hint = hints[index, :, RIGHT_FOOT_SMPL, :]
        frames = np.flatnonzero((np.linalg.norm(hint, axis=1) > 1e-6) & (np.arange(len(hint)) > 0))
        frame_indices = torch.as_tensor(frames - 1, dtype=torch.long)
        targets = torch.as_tensor(hint[frames] @ ROTATE_TO_G1.T, dtype=torch.float32)
        if args.mode == "window_smooth":
            frame_axis = torch.arange(q0.shape[0], dtype=torch.float32)
            begin, end = int(frame_indices.min()), int(frame_indices.max())
            ramp_up = torch.clamp((frame_axis - (begin - 20)) / 20, 0, 1)
            ramp_down = torch.clamp(((end + 20) - frame_axis) / 20, 0, 1)
            smootherstep = lambda value: value * value * (3 - 2 * value)
            envelope = smootherstep(ramp_up) * smootherstep(ramp_down)
            parameter = torch.nn.Parameter(torch.zeros(len(leg_indices), dtype=torch.float32))
            def expanded_delta() -> torch.Tensor:
                if args.max_joint_change_rad > 0:
                    # Cap changes so the optimizer cannot win only by replacing
                    # the kick with an extreme right-leg pose.
                    amplitude = args.max_joint_change_rad * torch.tanh(parameter)
                else:
                    amplitude = parameter
                return envelope[:, None] * amplitude[None, :]
        else:
            parameter = torch.nn.Parameter(torch.zeros((q0.shape[0], len(leg_indices)), dtype=torch.float32))
            def expanded_delta() -> torch.Tensor:
                return parameter
        optimizer = torch.optim.Adam([parameter], lr=0.08)

        def ankle_world() -> torch.Tensor:
            joints = q0.clone()
            joints[:, leg_indices] = torch.clamp(q0[:, leg_indices] + expanded_delta(), lower, upper)
            fk = chain.forward_kinematics({name: joints[:, j] for j, name in enumerate(joint_names)})
            local = fk["right_ankle_roll_link"].get_matrix()[:, :3, 3]
            return torch.einsum("tij,tj->ti", rotation, local) + root

        with torch.no_grad():
            before = float(torch.linalg.norm(ankle_world()[frame_indices] - targets, dim=1).mean())
        for _ in range(args.steps):
            optimizer.zero_grad()
            feet = ankle_world()[frame_indices]
            tracking = torch.square(feet - targets).sum(dim=1).mean()
            correction = expanded_delta()
            pose = torch.square(correction).mean()
            smooth = torch.square(correction[1:] - correction[:-1]).mean()
            loss = tracking + 0.01 * pose + 0.05 * smooth
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = float(torch.linalg.norm(ankle_world()[frame_indices] - targets, dim=1).mean())
            corrected_leg = torch.clamp(q0[:, leg_indices] + expanded_delta(), lower, upper)
            max_change = float(torch.abs(corrected_leg - q0[:, leg_indices]).max())
            max_speed_change = float(torch.abs((corrected_leg[1:] - corrected_leg[:-1]) - (q0[1:, leg_indices] - q0[:-1, leg_indices])).max())
            limit_violations = int(((corrected_leg < lower - 1e-4) | (corrected_leg > upper + 1e-4)).sum())
            corrected = q0.clone()
            corrected[:, leg_indices] = corrected_leg
        exported = None
        if export_dir is not None:
            destination = export_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace corrected reference: {destination}")
            revised = dict(robot)
            revised["joints"] = corrected.cpu().numpy().astype(np.asarray(robot["joints"]).dtype, copy=False)
            with destination.open("xb") as stream:
                pickle.dump(revised, stream, protocol=pickle.HIGHEST_PROTOCOL)
            exported = {"name": destination.name, "sha256": sha256(destination)}
        rows.append({
            "sample": index,
            "source_sha256": sha256(path),
            "before_error_m": before,
            "after_error_m": after,
            "max_joint_change_rad": max_change,
            "max_speed_change_rad_per_frame": max_speed_change,
            "joint_limit_violations": limit_violations,
            "support_leg_and_root_unchanged_by_design": True,
            "optimization_wall_time_s": time.perf_counter() - started,
            "exported_reference": exported,
        })
        print(f"sample {index}: {before:.3f} -> {after:.3f} m", flush=True)
    result = {
        "status": "component_pilot_only",
        "method": "right-leg constrained IK on raw G1 retarget references; fixed human motion",
        "parameterization": args.mode,
        "max_joint_change_cap_rad": args.max_joint_change_rad if args.mode == "window_smooth" else None,
        "external_target": "saved pre-generation right-foot hint, never robot kick_pos",
        "sample_selection": "first N available raw retarget files sorted by integer id; no tuning on other samples",
        "steps": args.steps,
        "source_npy_sha256": sha256(source),
        "mean_before_error_m": float(np.mean([r["before_error_m"] for r in rows])),
        "mean_after_error_m": float(np.mean([r["after_error_m"] for r in rows])),
        "rows": rows,
        "limitations": [
            "kinematics only: no contact forces, collisions, dynamics, semantics, or RL execution measured",
            "right ankle link is a proxy for the SMPL right-foot joint",
            "initial hyperparameters and sample size are exploratory, not held-out I1 Gate evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
