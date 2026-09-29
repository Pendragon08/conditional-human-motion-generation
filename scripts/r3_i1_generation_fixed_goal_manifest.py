"""Freeze the same pre-generation Kick scene goals for paired robot evaluation.

Read-only coordinate mapping. This neither solves IK nor modifies any motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch


ROTATE_TO_G1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = np.load(args.source, allow_pickle=True).item()
    human = np.asarray(source["motion"]).transpose(0, 3, 1, 2) @ ROTATE_TO_G1.T
    hints = np.asarray(source["hint"]) @ ROTATE_TO_G1.T
    chain = pk.build_chain_from_urdf(args.urdf.read_text())
    neutral = chain.forward_kinematics({name: torch.zeros(1) for name in chain.get_joint_parameter_names()})
    xyz = {name: neutral[name].get_matrix()[0, :3, 3].numpy()
           for name in ("right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link")}
    robot_leg = float(np.linalg.norm(xyz["right_hip_pitch_link"] - xyz["right_knee_link"])
                      + np.linalg.norm(xyz["right_knee_link"] - xyz["right_ankle_roll_link"]))
    paths = sorted(args.raw_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    rows = []
    for path in paths:
        sample = int(path.stem)
        with path.open("rb") as stream:
            robot = pickle.load(stream)
        root = np.asarray(robot["global_position"])
        constrained = np.flatnonzero(np.linalg.norm(hints[sample, 1:, 11], axis=-1) > 1e-6)
        if len(constrained) == 0:
            raise ValueError(f"Missing source foot hint for {sample}")
        anchor = int(constrained[len(constrained) // 2])
        human_frame = human[sample, anchor + 1]
        human_leg = np.linalg.norm(human_frame[2] - human_frame[5]) + np.linalg.norm(
            human_frame[5] - human_frame[8])
        scale = robot_leg / human_leg
        target = root[anchor] + scale * (hints[sample, anchor + 1, 11] - human_frame[0])
        rows.append({
            "sample": sample,
            "mapped_fixed_goal_xyz_m": np.asarray(target).tolist(),
            "goal_anchor_frame": anchor,
            "scale_robot_leg_over_human_leg": float(scale),
        })
    result = {
        "status": "read_only_fixed_scene_goal_manifest",
        "target_is_fixed_per_motion": True,
        "source_npy_sha256": digest(args.source),
        "raw_retarget_count": len(paths),
        "goal_from_pre_generation_hint": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
