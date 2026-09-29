"""Paired kinematic proxies for Kick event retention and ground interaction.

These are deliberately not text-motion R-precision, collision forces, or a
physics-simulator contact score. The human generation is identical in each
pair; only the retargeted robot right leg was corrected.
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link_world(chain, joint_names, robot, name: str) -> np.ndarray:
    q = torch.as_tensor(np.asarray(robot["joints"]), dtype=torch.float32)
    local = chain.forward_kinematics({joint: q[:, i] for i, joint in enumerate(joint_names)})[
        name
    ].get_matrix()[:, :3, 3].detach().numpy()
    rotation = Rotation.from_quat(np.roll(np.asarray(robot["global_pose"].rotation().wxyz), -1, axis=1))
    return rotation.apply(local) + np.asarray(robot["global_position"])


def event_features(right: np.ndarray, left: np.ndarray, root: np.ndarray, frames: np.ndarray) -> dict:
    right_relative = right - root
    left_relative = left - root
    before = np.arange(max(0, int(frames[0]) - 12), int(frames[0]))
    if not len(before):
        before = np.arange(min(10, len(right)))
    right_origin = np.median(right_relative[before], axis=0)
    left_origin = np.median(left_relative[before], axis=0)
    right_excursion = np.linalg.norm(right_relative[frames] - right_origin, axis=1)
    left_excursion = np.linalg.norm(left_relative[frames] - left_origin, axis=1)
    return {
        "right_excursion_m": float(right_excursion.max()),
        "left_excursion_m": float(left_excursion.max()),
        "right_foot_lift_m": float(right_relative[frames, 2].max() - right_origin[2]),
        "right_peak_frame": int(frames[int(right_excursion.argmax())]),
    }


def ground_proxy(foot: np.ndarray) -> dict:
    # Ankle link origin is not the sole geometry. These are only geometry proxies.
    z = foot[:, 2]
    low = z < float(np.quantile(z, 0.1) + 0.03)
    speed_xy = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=1)
    paired_low = low[1:] & low[:-1]
    return {
        "ankle_origin_below_ground_frame_fraction": float(np.mean(z < 0.0)),
        "low_ankle_frame_fraction": float(np.mean(low)),
        "low_ankle_horizontal_step_m": float(np.mean(speed_xy[paired_low])) if np.any(paired_low) else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, type=Path)
    p.add_argument("--corrected-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    stage = a.stage.resolve()
    corrected_dir = a.corrected_dir.resolve()
    source = stage / "Kick_sim/results.npy"
    hints = np.asarray(np.load(source, allow_pickle=True).item()["hint"])
    retarget_source = stage / "Kick_sim/retarget.py"
    joint_names = ast.literal_eval(re.search(r"joint_names = (\[.*?\])", retarget_source.read_text(), re.S).group(1))
    urdf = stage.parents[1] / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    chain = pk.build_chain_from_urdf(urdf.read_text())
    rows = []
    for raw_path in sorted((stage / "Kick_sim1").glob("*.pkl"), key=lambda x: int(x.stem)):
        corrected_path = corrected_dir / raw_path.name
        if not corrected_path.is_file():
            raise FileNotFoundError(corrected_path)
        with raw_path.open("rb") as f:
            raw = pickle.load(f)
        with corrected_path.open("rb") as f:
            changed = pickle.load(f)
        root = np.asarray(raw["global_position"])
        if not np.allclose(root, np.asarray(changed["global_position"]), atol=0, rtol=0):
            raise ValueError(f"Root changed: {raw_path.name}")
        index = int(raw_path.stem)
        hint = hints[index, :, 11, :]
        frames = np.flatnonzero((np.linalg.norm(hint, axis=1) > 1e-6) & (np.arange(len(hint)) > 0)) - 1
        if not len(frames):
            raise ValueError(f"No saved input hint: {index}")
        raw_right = link_world(chain, joint_names, raw, "right_ankle_roll_link")
        raw_left = link_world(chain, joint_names, raw, "left_ankle_roll_link")
        new_right = link_world(chain, joint_names, changed, "right_ankle_roll_link")
        new_left = link_world(chain, joint_names, changed, "left_ankle_roll_link")
        if not np.allclose(raw_left, new_left, atol=1e-5):
            raise ValueError(f"Support foot changed: {index}")
        original = event_features(raw_right, raw_left, root, frames)
        corrected = event_features(new_right, new_left, root, frames)
        original_event = original["right_excursion_m"] >= 0.12 and original["right_excursion_m"] >= 1.25 * original["left_excursion_m"]
        retained = (corrected["right_excursion_m"] >= 0.8 * original["right_excursion_m"]
                    and abs(corrected["right_peak_frame"] - original["right_peak_frame"]) <= 5)
        rows.append({
            "sample": index, "raw_sha256": digest(raw_path), "corrected_sha256": digest(corrected_path),
            "raw_event": original, "corrected_event": corrected,
            "raw_kick_event_proxy": bool(original_event),
            "event_amplitude_and_peak_retained": bool(retained),
            "raw_right_ground_proxy": ground_proxy(raw_right),
            "corrected_right_ground_proxy": ground_proxy(new_right),
            "support_foot_unchanged": True, "root_unchanged": True,
        })
        if len(rows) % 10 == 0:
            print(f"paired {len(rows)}", flush=True)
    result = {
        "status": "paired_kinematic_proxies_only",
        "source_npy_sha256": digest(source),
        "total_generated": len(hints), "paired": len(rows),
        "raw_kick_event_proxy_count": sum(r["raw_kick_event_proxy"] for r in rows),
        "event_amplitude_and_peak_retained_count": sum(r["event_amplitude_and_peak_retained"] for r in rows),
        "left_support_unchanged_count": sum(r["support_foot_unchanged"] for r in rows),
        "right_ankle_below_ground_frame_fraction_raw_mean": float(np.mean([r["raw_right_ground_proxy"]["ankle_origin_below_ground_frame_fraction"] for r in rows])),
        "right_ankle_below_ground_frame_fraction_corrected_mean": float(np.mean([r["corrected_right_ground_proxy"]["ankle_origin_below_ground_frame_fraction"] for r in rows])),
        "definition": "Right-foot root-relative excursion >=0.12m and >=1.25x left-foot; retention >=80% raw excursion with peak shift <=5 frames. Fixed exploratory thresholds, not a language evaluator.",
        "limitations": ["One Kick caption; no R-precision or cross-text semantic test.",
                        "Ankle origin is not foot sole mesh; low-ankle motion and below-ground tests are proxies, not contact force or collision proof.",
                        "The generated human motion is identical in raw/corrected pairs by construction."],
        "rows": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
