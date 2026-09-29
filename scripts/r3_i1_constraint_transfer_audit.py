"""Audit whether fixed human-space kick constraints survive G1 retargeting.

This is a read-only baseline diagnostic, not an I1 intervention or PPO result.
The target comes from OmniControl's saved input hint, never from its output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


ROTATE_TO_G1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
RIGHT_FOOT_SMPL = 11
THRESHOLD_M = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_m": None, "median_m": None, "p90_m": None, "success_at_15cm": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_m": float(arr.mean()),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.quantile(arr, 0.9)),
        "success_at_15cm": float(np.mean(arr <= THRESHOLD_M)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--constraint-source", type=Path,
                        help="Use this pre-generation baseline hint for every paired arm")
    args = parser.parse_args()
    stage = args.stage.resolve()
    source = (stage / "Kick_sim" / "results.npy").resolve()
    urdf = stage.parents[1] / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    retarget_source = stage / "Kick_sim/retarget.py"
    joint_source = retarget_source.read_text()
    joint_names = ast.literal_eval(re.search(r"joint_names = (\[.*?\])", joint_source, re.S).group(1))
    chain = pk.build_chain_from_urdf(urdf.read_text())
    data = np.load(source, allow_pickle=True).item()
    motions = np.asarray(data["motion"]).transpose(0, 3, 1, 2)
    if args.constraint_source:
        constraint_data = np.load(args.constraint_source, allow_pickle=True).item()
        if constraint_data["text"] != data["text"]:
            raise ValueError("Constraint-source text differs from evaluated motions")
        hints = np.asarray(constraint_data["hint"])
        if hints.shape != np.asarray(data["hint"]).shape:
            raise ValueError("Constraint-source hint shape differs")
    else:
        hints = np.asarray(data["hint"])
    rows = []
    example_count = 0
    for index, motion in enumerate(motions):
        target = hints[index, :, RIGHT_FOOT_SMPL, :] @ ROTATE_TO_G1.T
        human = motion @ ROTATE_TO_G1.T
        mask = np.linalg.norm(hints[index, :, RIGHT_FOOT_SMPL, :], axis=1) > 1e-6
        frames = np.flatnonzero(mask & (np.arange(len(mask)) > 0))
        if not len(frames):
            raise ValueError(f"Sample {index} has no fixed external kick hint")
        human_error = np.linalg.norm(human[frames, RIGHT_FOOT_SMPL] - target[frames], axis=1)
        robot_path = stage / "Kick_sim1" / f"{index}.pkl"
        row = {
            "sample": index,
            "constraint_frames": int(len(frames)),
            "human_mean_error_m": float(human_error.mean()),
            "human_p90_error_m": float(np.quantile(human_error, 0.9)),
            "retargeted": robot_path.exists(),
        }
        if robot_path.exists():
            with robot_path.open("rb") as stream:
                robot = pickle.load(stream)
            q = torch.as_tensor(np.asarray(robot["joints"]), dtype=torch.float32)
            transform = chain.forward_kinematics({name: q[:, j] for j, name in enumerate(joint_names)})
            ankle_local = transform["right_ankle_roll_link"].get_matrix()[:, :3, 3].detach().numpy()
            quaternion = np.asarray(robot["global_pose"].rotation().wxyz)
            ankle_world = Rotation.from_quat(np.roll(quaternion, -1, axis=1)).apply(ankle_local)
            root = np.asarray(robot["global_position"])
            ankle_world += root
            # The official retargeter discards the first generated frame.
            robot_frames = frames - 1
            robot_error = np.linalg.norm(ankle_world[robot_frames] - target[frames], axis=1)
            bias = (ankle_world[robot_frames] - target[frames]).mean(axis=0)
            translation_aligned_error = np.linalg.norm(
                ankle_world[robot_frames] - bias - target[frames], axis=1
            )
            row["robot_mean_error_m"] = float(robot_error.mean())
            row["robot_p90_error_m"] = float(np.quantile(robot_error, 0.9))
            row["robot_minus_human_mean_error_m"] = float(robot_error.mean() - human_error.mean())
            row["robot_root_relative_ankle_height_m"] = float(np.mean((ankle_world - root)[robot_frames, 2]))
            row["robot_mean_bias_xyz_m"] = bias.tolist()
            row["robot_translation_aligned_mean_error_m"] = float(translation_aligned_error.mean())
            root_relative_error = np.linalg.norm(
                (ankle_world - root)[robot_frames] - (target[frames] - human[frames, 0]),
                axis=1,
            )
            row["robot_root_relative_mean_error_m"] = float(root_relative_error.mean())
            if example_count < 3:
                t = int(frames[len(frames) // 2])
                row["example_middle_frame"] = {
                    "t": t,
                    "external_target_xyz_m": target[t].tolist(),
                    "human_foot_xyz_m": human[t, RIGHT_FOOT_SMPL].tolist(),
                    "robot_ankle_xyz_m": ankle_world[t - 1].tolist(),
                    "robot_root_xyz_m": root[t - 1].tolist(),
                }
                example_count += 1
        rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(motions):
            print(f"audited {index + 1}/{len(motions)} motions", flush=True)
    paired = [row for row in rows if row["retargeted"]]
    result = {
        "status": "observational_baseline_only",
        "source_npy_sha256": sha256(source),
        "constraint_source_sha256": sha256(args.constraint_source) if args.constraint_source else sha256(source),
        "retarget_source_sha256": sha256(retarget_source),
        "target_origin": "saved pre-generation right-foot hint, not output-derived kick_pos/grab_pos",
        "coordinate_map": "source xyz -> G1 (z,x,y), same as official retarget.py",
        "robot_alignment": "source frame t maps to G1 raw retarget frame t-1",
        "measurement": "mean 3D distance over the constrained frames; G1 right ankle link is a morphology-dependent proxy for SMPL right-foot joint",
        "threshold_m": THRESHOLD_M,
        "all_human": summarize([row["human_mean_error_m"] for row in rows]),
        "paired_human": summarize([row["human_mean_error_m"] for row in paired]),
        "paired_robot": summarize([row["robot_mean_error_m"] for row in paired]),
        "paired_robot_minus_human": summarize([row["robot_minus_human_mean_error_m"] for row in paired]),
        "paired_robot_translation_aligned": summarize(
            [row["robot_translation_aligned_mean_error_m"] for row in paired]
        ),
        "paired_robot_root_relative": summarize(
            [row["robot_root_relative_mean_error_m"] for row in paired]
        ),
        "mean_robot_bias_xyz_m": np.mean(
            [row["robot_mean_bias_xyz_m"] for row in paired], axis=0
        ).tolist() if paired else None,
        "not_retargeted": len(rows) - len(paired),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
