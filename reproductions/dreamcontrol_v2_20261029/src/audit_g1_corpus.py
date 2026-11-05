"""Read-only audit of the existing trusted ReactiveBFM G1 36D corpus.

This does not claim to produce the 263D DreamControl-v2 representation.
Pickle is executable: use only with a trusted corpus controlled by the user.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root: Path, fps: float, waist_tolerance: float) -> dict:
    data_path = root / "full_train.pkl"
    with data_path.open("rb") as f:
        corpus = pickle.load(f)
    if not isinstance(corpus, dict) or not isinstance(corpus.get("data_dict"), dict):
        raise ValueError("expected trusted corpus dict with data_dict")
    entries = corpus["data_dict"]
    splits = {}
    for split in ("train", "val", "test"):
        p = root / (split + ".txt")
        if p.exists():
            lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != len(set(lines)):
                raise ValueError(f"duplicate sample identifiers in {p}")
            splits[split] = set(lines)
    for name, ids in splits.items():
        missing = ids - entries.keys()
        if missing:
            raise ValueError(f"{name}: {len(missing)} IDs absent from corpus")
    split_names = list(splits)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            if splits[first] & splits[second]:
                raise ValueError(f"sample overlap between {first} and {second}")

    frames = 0
    invalid = []
    max_waist = 0.0
    waist_exceed = 0
    quat_errors = []
    captioned = 0
    lengths = []
    for key, entry in entries.items():
        motion = np.asarray(entry.get("motion")) if isinstance(entry, dict) else np.empty((0,))
        if motion.ndim != 2 or motion.shape[1] != 36 or motion.shape[0] < 2 or not np.isfinite(motion).all():
            invalid.append(str(key))
            continue
        if entry.get("length") is not None and int(entry["length"]) != len(motion):
            invalid.append(str(key))
            continue
        frames += len(motion)
        lengths.append(len(motion))
        captions = entry.get("text")
        captioned += int(bool(captions))
        quat_errors.append(float(np.max(np.abs(np.linalg.norm(motion[:, 3:7], axis=1) - 1.0))))
        # 36D is xyz + xyzw + 29 actuators. The source waist roll and pitch
        # are actuator indices 13 and 14; retaining only 27 DoF is not valid
        # when these channels move, unless the motion is re-optimized.
        waist = float(np.max(np.abs(motion[:, [20, 21]])))
        max_waist = max(max_waist, waist)
        waist_exceed += int(waist > waist_tolerance)

    if fps <= 0 or waist_tolerance < 0:
        raise ValueError("fps must be positive and waist tolerance non-negative")
    return {
        "source": str(data_path), "source_sha256": sha256(data_path),
        "declared_fps": fps, "target_fps": 20,
        "sample_count": len(entries), "valid_structure_count": len(lengths),
        "invalid_count": len(invalid), "invalid_examples": invalid[:20],
        "frame_count": frames, "captioned_count": captioned,
        "length_frames_min": min(lengths) if lengths else None,
        "length_frames_max": max(lengths) if lengths else None,
        "quaternion_max_abs_norm_error": max(quat_errors) if quat_errors else None,
        "waist_roll_pitch_max_abs_rad": max_waist,
        "waist_tolerance_rad": waist_tolerance,
        "clips_exceeding_waist_tolerance": waist_exceed,
        "splits": {name: len(ids) for name, ids in splits.items()},
        "warning": "36D is 3 xyz + 4 xyzw + 29 actuators, not the paper's 263D encoding; an FK, coordinate, 27-DoF and representation audit is required.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir", type=Path)
    p.add_argument("--source-fps", type=float, required=True)
    p.add_argument("--waist-tolerance-rad", type=float, default=0.01)
    args = p.parse_args()
    print(json.dumps(audit(args.corpus_dir, args.source_fps, args.waist_tolerance_rad), indent=2))


if __name__ == "__main__":
    main()
