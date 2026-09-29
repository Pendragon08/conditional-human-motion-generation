"""Audit reference temporal changes after I1 right-leg correction.

This is a kinematic diagnostic, not a motor torque or physics feasibility test.
DreamControl's Kick MotionLib uses 2/20 seconds per reference frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


RIGHT_LEG = slice(6, 12)
REFERENCE_DT_S = 2.0 / 20.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_paths = sorted(args.raw_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    corrected_paths = sorted(args.corrected_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    if not raw_paths or [p.name for p in raw_paths] != [p.name for p in corrected_paths]:
        raise ValueError("Raw and corrected file sets differ")
    rows = []
    for original_path, changed_path in zip(raw_paths, corrected_paths):
        with original_path.open("rb") as stream:
            original = pickle.load(stream)
        with changed_path.open("rb") as stream:
            changed = pickle.load(stream)
        if set(original) != set(changed):
            raise ValueError(f"Different fields in {original_path.name}")
        q0 = np.asarray(original["joints"], dtype=np.float64)
        q1 = np.asarray(changed["joints"], dtype=np.float64)
        if q0.shape != q1.shape or q0.ndim != 2 or q0.shape[0] < 3:
            raise ValueError(f"Invalid joint arrays in {original_path.name}")
        if not np.array_equal(q0[:, :6], q1[:, :6]) or not np.array_equal(q0[:, 12:], q1[:, 12:]):
            raise ValueError(f"Non-right-leg joints changed in {original_path.name}")
        if not np.array_equal(np.asarray(original["global_position"]), np.asarray(changed["global_position"])):
            raise ValueError(f"Root trajectory changed in {original_path.name}")
        base_v = np.diff(q0[:, RIGHT_LEG], axis=0) / REFERENCE_DT_S
        new_v = np.diff(q1[:, RIGHT_LEG], axis=0) / REFERENCE_DT_S
        base_a = np.diff(base_v, axis=0) / REFERENCE_DT_S
        new_a = np.diff(new_v, axis=0) / REFERENCE_DT_S
        rows.append({
            "sample": int(original_path.stem),
            "raw_sha256": digest(original_path),
            "corrected_sha256": digest(changed_path),
            "raw_peak_abs_right_leg_velocity_rad_s": float(np.abs(base_v).max()),
            "corrected_peak_abs_right_leg_velocity_rad_s": float(np.abs(new_v).max()),
            "raw_peak_abs_right_leg_acceleration_rad_s2": float(np.abs(base_a).max()),
            "corrected_peak_abs_right_leg_acceleration_rad_s2": float(np.abs(new_a).max()),
            "peak_added_abs_velocity_rad_s": float(np.abs(new_v - base_v).max()),
            "peak_added_abs_acceleration_rad_s2": float(np.abs(new_a - base_a).max()),
        })
    result = {
        "status": "reference_temporal_kinematics_only",
        "raw_directory": str(args.raw_dir.resolve()),
        "corrected_directory": str(args.corrected_dir.resolve()),
        "reference_dt_s": REFERENCE_DT_S,
        "paired_count": len(rows),
        "mean_peak_added_abs_velocity_rad_s": float(np.mean([row["peak_added_abs_velocity_rad_s"] for row in rows])),
        "max_peak_added_abs_velocity_rad_s": float(np.max([row["peak_added_abs_velocity_rad_s"] for row in rows])),
        "mean_peak_added_abs_acceleration_rad_s2": float(np.mean([row["peak_added_abs_acceleration_rad_s2"] for row in rows])),
        "max_peak_added_abs_acceleration_rad_s2": float(np.max([row["peak_added_abs_acceleration_rad_s2"] for row in rows])),
        "limitations": [
            "Finite differences at the reference sample rate; no motor torque, thermal or hardware safety inference.",
            "The Isaac actuator configuration does not enforce its legacy velocity_limit field.",
        ],
        "rows": rows,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
