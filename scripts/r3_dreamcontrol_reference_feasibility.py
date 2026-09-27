"""Read-only check of DreamControl Kick reference/goal geometry.

Runs with the official Isaac Sim Python and the project's DreamControl overlay.
This does not evaluate a learned policy or modify training data.
"""

import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DreamControl"
URDF = ROOT / "Training" / "HumanoidVerse" / "humanoidverse" / "data" / "robots" / "g1" / "g1_27dof.urdf"
DATA = ROOT / "TrajGen" / "sample" / "Kick_sim2"
JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main():
    chain = pk.build_chain_from_urdf(URDF.read_text())
    rows = []
    link_name = None
    for path in sorted(glob.glob(str(DATA / "*.pkl"))):
        with open(path, "rb") as stream:
            data = pickle.load(stream)
        joints = torch.as_tensor(np.asarray(data["joints"])[:200], dtype=torch.float32)
        q_dict = {name: joints[:, i] for i, name in enumerate(JOINTS)}
        transforms = chain.forward_kinematics(q_dict)
        names = list(transforms)
        link_name = names[13]
        local = transforms[link_name].get_matrix()[:, :3, 3].detach().numpy()
        wxyz = np.asarray(data["global_pose"].rotation().wxyz)[:200]
        xyzw = np.roll(wxyz, -1, axis=-1)
        root = np.asarray(data["global_position"])[:200].copy()
        root[:, 2] += 0.035
        world = Rotation.from_quat(xyzw).apply(local) + root
        idx = int(np.argmax(world[:, 0]))
        # Kick's touch_goal uses motion_lib.grab_pos + offsets as its goal.
        # For Kick, grab_pos is the reference right ankle position at max x.
        # Do not substitute reset_object_state's default height (0.67 m):
        # that reset is not used by the Kick success metric.
        goal = world[idx].copy()
        window = world[max(0, idx - 3): min(len(world), idx + 4)]
        rows.append({
            "file": Path(path).name,
            "strike_index": idx,
            "reference_foot_z_at_strike": float(world[idx, 2]),
            "reference_distance_at_strike_m": float(np.linalg.norm(world[idx] - goal)),
            "minimum_reference_distance_near_strike_m": float(np.linalg.norm(window - goal, axis=1).min()),
        })
    near = np.array([row["minimum_reference_distance_near_strike_m"] for row in rows])
    result = {
        "reference_clips": len(rows),
        "fk_index_13_link": link_name,
        "goal_definition": "reference right ankle position at maximum forward reach; no fixed target height",
        "touch_threshold_m": 0.1,
        "clips_reference_near_strike_below_threshold": int((near < 0.1).sum()),
        "examples": rows[:5],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
