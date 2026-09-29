"""Summarize the untouched 56-sample set after the 8-sample IK pilot."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--all", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text())
    all_results = json.loads(args.all.read_text())
    ids = {row["sample"] for row in pilot["rows"]}
    holdout = [row for row in all_results["rows"] if row["sample"] not in ids]
    if len(ids) != 8 or len(holdout) != 56:
        raise RuntimeError("Expected eight pilot and 56 held-out retargeted motions")
    before = [row["before_error_m"] for row in holdout]
    after = [row["after_error_m"] for row in holdout]
    success = [row["after_error_m"] <= 0.15 for row in holdout]
    low_speed_change = [row["max_speed_change_rad_per_frame"] <= 0.1 for row in holdout]
    result = {
        "status": "heldout_kinematic_component_result_not_full_I1_Gate",
        "pilot_ids": sorted(ids),
        "heldout_ids": [row["sample"] for row in holdout],
        "heldout_count": len(holdout),
        "before_mean_error_m": mean(before),
        "after_mean_error_m": mean(after),
        "before_median_error_m": median(before),
        "after_median_error_m": median(after),
        "relative_mean_error_reduction": 1 - mean(after) / mean(before),
        "improved_count": sum(a < b for a, b in zip(after, before)),
        "condition_success_threshold_m": 0.15,
        "condition_success_count": sum(success),
        "diagnostic_speed_change_threshold_rad_per_frame": 0.1,
        "speed_change_above_diagnostic_threshold_count": sum(not x for x in low_speed_change),
        "condition_success_and_low_speed_change_count": sum(a and b for a, b in zip(success, low_speed_change)),
        "joint_limit_violation_count": sum(row["joint_limit_violations"] for row in holdout),
        "pilot_file_sha256": sha256(args.pilot),
        "all64_file_sha256": sha256(args.all),
        "caveats": [
            "0.1 rad/frame is a screening heuristic, not a validated G1 torque/velocity limit",
            "only Kick, one generation seed, and 56 available retarget outputs are evaluated",
            "no collisions, contact forces, dynamics, semantic score, or RL execution measured",
            "human-space motion is held fixed; this alone does not validate an improved diffusion generator",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
