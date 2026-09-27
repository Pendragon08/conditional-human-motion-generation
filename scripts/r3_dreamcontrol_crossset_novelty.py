"""Nearest-neighbor overlap of candidate G1 references with supplied references."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load(folder):
    names, joints, roots = [], [], []
    for path in sorted(folder.glob("*.pkl")):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        names.append(path.name)
        joints.append(np.asarray(data["joints"], dtype=np.float32))
        roots.append(np.asarray(data["global_position"], dtype=np.float32))
    return names, np.stack(joints), np.stack(roots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplied", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old_names, old_q, old_root = load(args.supplied)
    new_names, new_q, new_root = load(args.candidate)
    assert old_q.shape[1:] == new_q.shape[1:]
    rows = []
    for name, q, root in zip(new_names, new_q, new_root):
        q_rmse = np.sqrt(np.mean((old_q - q) ** 2, axis=(1, 2)))
        r_rmse = np.sqrt(np.mean((old_root - root) ** 2, axis=(1, 2)))
        nearest = int(np.argmin(q_rmse))
        rows.append({
            "candidate": name,
            "nearest_supplied": old_names[nearest],
            "nearest_joint_rmse_rad": float(q_rmse[nearest]),
            "nearest_root_rmse_m": float(r_rmse[nearest]),
            "same_name": name == old_names[nearest],
        })
    joints = np.array([row["nearest_joint_rmse_rad"] for row in rows])
    roots = np.array([row["nearest_root_rmse_m"] for row in rows])
    summary = {
        "supplied_count": len(old_names),
        "candidate_count": len(new_names),
        "candidate_nearest_joint_rmse_median_rad": float(np.median(joints)),
        "candidate_nearest_root_rmse_median_m": float(np.median(roots)),
        "near_duplicate_count_0p01rad_1mm": int(np.sum((joints < 0.01) & (roots < 0.001))),
        "near_duplicate_count_0p02rad_2mm": int(np.sum((joints < 0.02) & (roots < 0.002))),
        "nearest_same_name_count": sum(row["same_name"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
