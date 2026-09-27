"""Inspect generated G1 reference files without changing them."""

import argparse
import ast
import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stage = args.stage
    source = (stage / "Kick_sim" / "retarget.py").read_text()
    names = ast.literal_eval(re.search(r"joint_names = (\[.*?\])", source, re.S).group(1))
    urdf = stage.parents[1] / "Training" / "HumanoidVerse" / "humanoidverse" / "data" / "robots" / "g1" / "g1_27dof.urdf"
    xml = ET.parse(urdf).getroot()
    limits = {}
    for joint in xml.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") and limit.get("upper"):
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    result = {"stage": str(stage), "sources": {}, "files": {}}
    for folder in ("Kick_sim1", "Kick_sim2"):
        rows = []
        for path in sorted((stage / folder).glob("*.pkl")):
            with path.open("rb") as stream:
                data = pickle.load(stream)
            q = np.asarray(data["joints"])
            xyz = np.asarray(data["global_position"])
            pose = np.asarray(data["global_pose"].wxyz_xyz)
            index = int(data["grab_idx"])
            violations = 0
            max_limit_excess = 0.0
            violating_names = []
            for j, name in enumerate(names):
                if name in limits:
                    lower, upper = limits[name]
                    excess = np.maximum(lower - q[:, j], q[:, j] - upper)
                    count = int(np.count_nonzero(excess > 1e-3))
                    violations += count
                    if count:
                        violating_names.append(name)
                        max_limit_excess = max(max_limit_excess, float(excess.max()))
            rows.append({
                "file": path.name,
                "frames": int(q.shape[0]),
                "joint_count": int(q.shape[1]),
                "finite": bool(np.isfinite(q).all() and np.isfinite(xyz).all() and np.isfinite(pose).all()),
                "root_height_min": float(xyz[:, 2].min()),
                "root_height_max": float(xyz[:, 2].max()),
                "strike_index_valid": bool(0 <= index < len(q)),
                "joint_limit_violations": violations,
                "max_joint_limit_excess_rad": max_limit_excess,
                "violating_joint_names": violating_names,
            })
        result["files"][folder] = rows
        result["sources"][folder] = {
            "count": len(rows),
            "all_finite": all(row["finite"] for row in rows),
            "all_strike_indices_valid": all(row["strike_index_valid"] for row in rows),
            "total_joint_limit_violations": sum(row["joint_limit_violations"] for row in rows),
            "frame_lengths": sorted(set(row["frames"] for row in rows)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["sources"], indent=2))


if __name__ == "__main__":
    main()
