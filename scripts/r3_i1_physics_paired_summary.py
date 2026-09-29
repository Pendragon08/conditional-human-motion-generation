"""Summarize fixed-controller Isaac rollout without conflating proximity and impact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "official_goal_proximity_success_percent",
    "joint_mae_rad",
    "root_mean_error_m",
    "early_resets_per_environment",
    "root_below_0_3_fraction",
    "left_ankle_contact_frame_fraction",
    "right_ankle_contact_frame_fraction",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    rows = []
    for seed in (42, 43, 44):
        paths = {arm: a.results / f"r3_i1_physics_seed{seed}_{arm}.json" for arm in ("raw", "corrected")}
        runs = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
        raw, corrected = runs["raw"], runs["corrected"]
        for run in (raw, corrected):
            if run["seed"] != 42 or run["num_envs"] != 256 or run["video_length"] != 500:
                raise ValueError(f"Unexpected evaluation setup: {paths}")
            if not run["contact_sensor_available"]:
                raise ValueError(f"Contact sensor missing: {paths}")
        if raw["checkpoint"] != corrected["checkpoint"]:
            raise ValueError(f"Policy mismatch: {paths}")
        row = {
            "generation_seed": seed,
            "same_initial_motion_ids": raw["initial_motion_ids"] == corrected["initial_motion_ids"],
            "same_motion_id_counts": raw["motion_id_counts"] == corrected["motion_id_counts"],
            "motion_library_size": raw["motion_library_size"],
            "policy_checkpoint": raw["checkpoint"],
            "raw_metrics_path": str(paths["raw"]),
            "corrected_metrics_path": str(paths["corrected"]),
        }
        for metric in METRICS:
            row[metric] = {arm: runs[arm][metric] for arm in runs}
        rows.append(row)
    result = {
        "status": "simulation_only_fixed_tracker_diagnostic",
        "rows": rows,
        "all_pairs_same_initial_motion_ids": all(r["same_initial_motion_ids"] for r in rows),
        "all_pairs_same_motion_id_counts": all(r["same_motion_id_counts"] for r in rows),
        "target_impact_contact_measured": False,
        "limitations": [
            "Kick goal success is geometric proximity, not impact force.",
            "Ankle contact sensors do not separate ground from target contact.",
            "Fixed controller success/failure measures execution under this policy; it is not a proof of mechanical feasibility or impossibility.",
            "Only the retargeted subsets are in the Isaac rollouts; original generation rejections are counted separately in I1 end-to-end rates.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
