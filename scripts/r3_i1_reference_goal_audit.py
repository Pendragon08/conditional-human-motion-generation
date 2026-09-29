"""Read-only G1 ankle-to-fixed-goal audit for paired reference motions."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--retarget-source", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ast.literal_eval(re.search(r"joint_names = (\[.*?\])",
                                  args.retarget_source.read_text(), re.S).group(1))
    chain = pk.build_chain_from_urdf(args.urdf.read_text())
    goal_rows = json.loads(args.goals.read_text())["rows"]
    result_rows = []
    for row in goal_rows:
        sample = int(row["sample"])
        target = np.asarray(row["mapped_fixed_goal_xyz_m"])
        by_arm = {}
        for arm, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
            with (directory / f"{sample}.pkl").open("rb") as stream:
                motion = pickle.load(stream)
            joints = torch.as_tensor(np.asarray(motion["joints"]), dtype=torch.float32)
            transforms = chain.forward_kinematics({name: joints[:, idx] for idx, name in enumerate(names)})
            foot_local = transforms["right_ankle_roll_link"].get_matrix()[:, :3, 3].detach().numpy()
            quat = np.asarray(motion["global_pose"].rotation().wxyz)
            foot_world = Rotation.from_quat(np.roll(quat, -1, axis=1)).apply(foot_local)
            foot_world += np.asarray(motion["global_position"])
            anchor = int(row["goal_anchor_frame"])
            window = slice(max(0, anchor - 8), min(len(foot_world), anchor + 9))
            error = np.linalg.norm(foot_world[window] - target, axis=-1)
            by_arm[arm] = {
                "mean_error_m": float(error.mean()),
                "minimum_error_m": float(error.min()),
                "anchor_error_m": float(np.linalg.norm(foot_world[anchor] - target)),
            }
        result_rows.append({"sample": sample, **by_arm})
    summary = {}
    for arm in ("baseline", "candidate"):
        summary[arm] = {key: float(np.mean([row[arm][key] for row in result_rows]))
                        for key in ("mean_error_m", "minimum_error_m", "anchor_error_m")}
        summary[arm]["minimum_le_0_1_m"] = sum(
            row[arm]["minimum_error_m"] <= 0.1 for row in result_rows)
    result = {
        "status": "read_only_reference_geometry_not_simulation",
        "same_fixed_pre_generation_goals": True,
        "count": len(result_rows),
        "summary": summary,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
