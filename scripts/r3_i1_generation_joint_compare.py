"""Paired human-motion diagnostic for the no-extra-IK generation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(motion: np.ndarray, original_hint: np.ndarray) -> list[dict]:
    right_mask = np.linalg.norm(original_hint[:, :, 11], axis=-1) > 1e-6
    rows = []
    for idx, one in enumerate(motion):
        sequence = one.transpose(2, 0, 1)
        foot = sequence[:, 11]
        left = sequence[:, 10]
        pelvis = sequence[:, 0]
        mask = right_mask[idx]
        kick_window = slice(17, 40)
        rows.append({
            "sample": idx,
            "original_right_foot_hint_error_m": float(np.linalg.norm(
                foot[mask] - original_hint[idx, mask, 11], axis=-1).mean()),
            "left_foot_stance_speed_m_per_frame": float(np.linalg.norm(
                np.diff(left[kick_window], axis=0), axis=-1).mean()),
            "left_foot_stance_drift_m": float(np.linalg.norm(
                left[39] - left[17])),
            "pelvis_drift_during_kick_m": float(np.linalg.norm(pelvis[39] - pelvis[17])),
            "whole_body_accel_proxy_m_per_frame2": float(np.linalg.norm(
                np.diff(sequence, n=2, axis=0), axis=-1).mean()),
            "right_foot_forward_peak_frame": int(np.argmax((foot - pelvis)[:, 2])),
            "finite": bool(np.isfinite(sequence).all()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="joint")
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()
    paths = {arm: args.base / arm / "results.npy" for arm in ("baseline", args.arm)}
    if args.baseline_file:
        paths["baseline"] = args.baseline_file
    if args.candidate_file:
        paths[args.arm] = args.candidate_file
    data = {arm: np.load(path, allow_pickle=True).item() for arm, path in paths.items()}
    original_hint = np.asarray(data["baseline"]["hint"])
    joint_hint = np.asarray(data[args.arm]["hint"])
    base_motion = np.asarray(data["baseline"]["motion"])
    joint_motion = np.asarray(data[args.arm]["motion"])
    if base_motion.shape != joint_motion.shape or base_motion.shape[0] != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} paired motions with matching shapes")
    if data["baseline"]["text"] != data[args.arm]["text"]:
        raise ValueError("Text prompts differ")
    if not np.array_equal(np.asarray(data["baseline"]["lengths"]),
                          np.asarray(data[args.arm]["lengths"])):
        raise ValueError("Motion lengths differ")
    base_right = original_hint[:, :, 11]
    joint_right = joint_hint[:, :, 11]
    # The new arm is allowed only two predeclared recovery keyframes.
    keep = np.ones(base_right.shape[1], dtype=bool)
    keep[[0, 65]] = False
    if not np.allclose(base_right[:, keep], joint_right[:, keep], atol=1e-6):
        raise ValueError("Original right-foot spatial goal was changed")
    if np.any(np.linalg.norm(joint_hint[:, :, 21], axis=-1) > 1e-6):
        raise ValueError("Right-wrist hint would change the official retargeter's task")
    if not np.array_equal(
        np.linalg.norm(original_hint[:, :, 11], axis=-1) > 1e-6,
        np.linalg.norm(joint_hint[:, :, 11], axis=-1) > 1e-6,
    ):
        # The only allowed extra right-foot hints are predefined recovery
        # frames; any other change is a protocol violation.
        changes = np.flatnonzero(np.any(
            (np.linalg.norm(original_hint[:, :, 11], axis=-1) > 1e-6)
            != (np.linalg.norm(joint_hint[:, :, 11], axis=-1) > 1e-6), axis=0
        ))
        if not set(changes.tolist()).issubset({0, 65}):
            raise ValueError(f"Unexpected right-foot hint frames: {changes.tolist()}")
    rows = {arm: metrics(np.asarray(data[arm]["motion"]), original_hint)
            for arm in paths}
    summary = {}
    for arm, arm_rows in rows.items():
        summary[arm] = {
            key: float(np.mean([row[key] for row in arm_rows]))
            for key in (
                "original_right_foot_hint_error_m",
                "left_foot_stance_speed_m_per_frame",
                "left_foot_stance_drift_m",
                "pelvis_drift_during_kick_m",
                "whole_body_accel_proxy_m_per_frame2",
            )
        }
        summary[arm]["finite_count"] = sum(row["finite"] for row in arm_rows)
        summary[arm]["right_foot_peak_in_17_to_39"] = sum(
            17 <= row["right_foot_forward_peak_frame"] <= 39 for row in arm_rows
        )
    report = {
        "status": "human_generation_audit_no_extra_ik",
        "sample_count": args.expected_count,
        "same_text_and_original_spatial_goal": True,
        "contact_and_timing_are_kinematic_proxies": True,
        "inputs_sha256": {arm: digest(path) for arm, path in paths.items()},
        "summary": summary,
        "rows": rows,
        "limitations": ["One Kick prompt", "No simulator or PPO result in this file"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
