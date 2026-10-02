"""Summarize fixed-scene whole-body refinement without dropping failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {42: 64, 43: 64, 44: 63}
    summaries = {}
    for seed, count in expected.items():
        path = args.results / f"r3_i1_whole_body_fixed_scene_seed{seed}.json"
        result = json.loads(path.read_text())
        if result.get("target_is_fixed_per_motion") is not True or len(result["rows"]) != count:
            raise ValueError(f"Bad fixed-scene result for seed {seed}")
        rows = result["rows"]
        if len({row["sample"] for row in rows}) != count:
            raise ValueError(f"Duplicate motion ID in seed {seed}")
        def values(key: str) -> np.ndarray:
            return np.asarray([row[key] for row in rows], dtype=np.float64)
        raw = values("raw_mapped_target_error_m")
        refined = values("refined_mapped_target_error_m")
        peak_shift = np.abs(values("peak_forward_frame_shift"))
        amplitude_ratio = values("kick_peak_forward_amplitude_ratio")
        left_drift = values("left_stance_drift_m")
        extra_speed = values("max_joint_speed_change_rad_per_frame")
        passes_proxy = (
            (refined <= 0.15) & (left_drift <= 0.03)
            & (peak_shift <= 5) & (amplitude_ratio >= 0.8)
            & (extra_speed <= 0.1)
        )
        summaries[str(seed)] = {
            "generated_inputs": 100,
            "official_retarget_accepted": count,
            "official_retarget_rejected": 100 - count,
            "raw_mapped_target_error_mean_m": float(raw.mean()),
            "refined_mapped_target_error_mean_m": float(refined.mean()),
            "improved_paired_motions": int((refined < raw).sum()),
            "raw_target_le_0_15_m_per_100": int((raw <= 0.15).sum()),
            "refined_target_le_0_15_m_per_100": int((refined <= 0.15).sum()),
            "refined_all_proxy_checks_per_100": int(passes_proxy.sum()),
            "left_stance_drift_mean_m": float(left_drift.mean()),
            "max_added_joint_delta_per_frame_rad": float(extra_speed.max()),
            "peak_shift_le_5_frames": int((peak_shift <= 5).sum()),
            "amplitude_retention_ge_0_8": int((amplitude_ratio >= 0.8).sum()),
            "joint_limit_violations": int(sum(row["joint_limit_violations"] for row in rows)),
            "optimization_time_mean_s": float(values("elapsed_wall_s").mean()),
        }
    output = {
        "status": "fixed_scene_kinematic_validation_only",
        "definition": "one fixed morphology-scaled pre-generation goal per motion; rejected retarget inputs remain in each 100-motion denominator",
        "thresholds_are_exploratory": True,
        "seeds": summaries,
        "limitations": [
            "No independent multi-text result, target collision force, or matched-budget PPO yet",
            "The seed42 pilot was used for weight selection; seeds43/44 are fixed-hyperparameter checks",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
