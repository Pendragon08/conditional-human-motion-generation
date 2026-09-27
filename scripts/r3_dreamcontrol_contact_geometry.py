"""Read-only foot/ground geometry audit for generated G1 references."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stage = args.stage
    source = (stage / "Kick_sim" / "retarget.py").read_text()
    names = ast.literal_eval(re.search(r"joint_names = (\[.*?\])", source, re.S).group(1))
    urdf = stage.parents[1] / "Training" / "HumanoidVerse" / "humanoidverse" / "data" / "robots" / "g1" / "g1_27dof.urdf"
    chain = pk.build_chain_from_urdf(urdf.read_text())
    rows = []
    for path in sorted((stage / "Kick_sim2").glob("*.pkl")):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        q = torch.as_tensor(np.asarray(data["joints"]), dtype=torch.float32)
        fk = chain.forward_kinematics({name: q[:, i] for i, name in enumerate(names)})
        rot = Rotation.from_quat(np.roll(np.asarray(data["global_pose"].rotation().wxyz), -1, axis=1))
        root = np.asarray(data["global_position"])
        feet = []
        for link in ("left_ankle_roll_link", "right_ankle_roll_link"):
            local = fk[link].get_matrix()[:, :3, 3].detach().numpy()
            feet.append(rot.apply(local) + root)
        feet = np.stack(feet, axis=1)
        support_z = feet[:, :, 2].min(axis=1)
        rows.append({
            "file": path.name,
            "support_height_min_m": float(support_z.min()),
            "support_height_max_m": float(support_z.max()),
            "frames_below_minus_2cm": int((support_z < -0.02).sum()),
            "frames_both_feet_above_10cm": int((support_z > 0.10).sum()),
            "frames": int(len(q)),
        })
    result = {
        "stage": str(stage),
        "clips": len(rows),
        "total_frames": sum(row["frames"] for row in rows),
        "frames_below_minus_2cm": sum(row["frames_below_minus_2cm"] for row in rows),
        "frames_both_feet_above_10cm": sum(row["frames_both_feet_above_10cm"] for row in rows),
        "minimum_support_height_m": min((row["support_height_min_m"] for row in rows), default=None),
        "rows": rows,
        "note": "FK geometry only; no contact forces or dynamic execution measured",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
