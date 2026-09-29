"""Verify exported G1 references change only the intended right-leg joint values."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


EXPECTED_FIELDS = {"global_pose", "global_position", "joints", "kick_pos", "grab_idx"}
RIGHT_LEG = slice(6, 12)


def sha256(path: Path) -> str:
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
    originals = sorted(args.raw_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    changed = sorted(args.corrected_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    if not originals or [p.name for p in originals] != [p.name for p in changed]:
        raise ValueError("Raw and corrected reference sets differ")
    rows = []
    for raw_path, edited_path in zip(originals, changed):
        with raw_path.open("rb") as stream:
            raw = pickle.load(stream)
        with edited_path.open("rb") as stream:
            edited = pickle.load(stream)
        if set(raw) != EXPECTED_FIELDS or set(edited) != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected reference fields in {raw_path.name}: {set(raw)} {set(edited)}")
        invariant_arrays = {
            "global_position": (raw["global_position"], edited["global_position"]),
            "global_pose.translation": (raw["global_pose"].translation(), edited["global_pose"].translation()),
            "global_pose.rotation": (raw["global_pose"].rotation().wxyz, edited["global_pose"].rotation().wxyz),
            "kick_pos": (raw["kick_pos"], edited["kick_pos"]),
            "grab_idx": (raw["grab_idx"], edited["grab_idx"]),
        }
        for label, (before, after) in invariant_arrays.items():
            if not np.array_equal(np.asarray(before), np.asarray(after), equal_nan=True):
                raise ValueError(f"{label} changed: {raw_path.name}")
        q0 = np.asarray(raw["joints"])
        q1 = np.asarray(edited["joints"])
        if q0.shape != q1.shape or q0.ndim != 2 or q0.shape[1] != 27:
            raise ValueError(f"Joint shape changed: {raw_path.name}")
        if not np.array_equal(q0[:, :RIGHT_LEG.start], q1[:, :RIGHT_LEG.start], equal_nan=True):
            raise ValueError(f"Left-leg joints changed: {raw_path.name}")
        if not np.array_equal(q0[:, RIGHT_LEG.stop:], q1[:, RIGHT_LEG.stop:], equal_nan=True):
            raise ValueError(f"Torso/arm joints changed: {raw_path.name}")
        rows.append({
            "sample": int(raw_path.stem),
            "raw_sha256": sha256(raw_path),
            "corrected_sha256": sha256(edited_path),
            "right_leg_max_abs_delta_rad": float(np.max(np.abs(q1[:, RIGHT_LEG] - q0[:, RIGHT_LEG]))),
        })
    result = {
        "status": "all_exported_fields_verified",
        "raw_dir": str(args.raw_dir.resolve()),
        "corrected_dir": str(args.corrected_dir.resolve()),
        "paired_count": len(rows),
        "unchanged_fields": sorted(EXPECTED_FIELDS - {"joints"}),
        "joint_columns_changed": list(range(RIGHT_LEG.start, RIGHT_LEG.stop)),
        "rows": rows,
        "note": "kick_pos is an unchanged task flag in this export; DreamControl MotionLib derives its runtime Kick goal from joint forward kinematics, so frozen-goal evaluation is separately required.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(rows)} reference pairs")


if __name__ == "__main__":
    main()
