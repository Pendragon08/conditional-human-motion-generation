"""Check and summarize I1 follow-up diagnostics without upgrading proxies to claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources: dict[str, str] = {}

    def load(name: str) -> dict:
        path = args.results / name
        sources[name] = sha256(path)
        return read(path)

    paired = []
    for seed, expected in ((42, 64), (43, 64), (44, 63)):
        raw = load(f"r3_i1_physics_seed{seed}_raw.json")
        corrected = load(f"r3_i1_physics_seed{seed}_corrected.json")
        semantic = load(f"r3_i1_semantic_contact_proxy_seed{seed}.json")
        temporal = load(f"r3_i1_temporal_quality_seed{seed}.json")
        for record in (raw, corrected):
            if (record["num_envs"], record["video_length"], record["seed"], record["motion_library_size"]) != (256, 500, 42, expected):
                raise ValueError(f"Unexpected Isaac setup seed={seed}")
            if not record["contact_sensor_available"]:
                raise ValueError(f"Missing contact sensor seed={seed}")
        if raw["checkpoint"] != corrected["checkpoint"] or raw["initial_motion_ids"] != corrected["initial_motion_ids"]:
            raise ValueError(f"Mismatched fixed-policy initial conditions seed={seed}")
        if semantic["paired"] != expected or temporal["paired_count"] != expected:
            raise ValueError(f"Mismatched reference count seed={seed}")
        paired.append({
            "generation_seed": seed,
            "retargeted_pairs": expected,
            "official_goal_proximity_success_percent_raw": raw["official_goal_proximity_success_percent"],
            "official_goal_proximity_success_percent_corrected": corrected["official_goal_proximity_success_percent"],
            "joint_mae_rad_raw": raw["joint_mae_rad"],
            "joint_mae_rad_corrected": corrected["joint_mae_rad"],
            "early_resets_per_environment_raw": raw["early_resets_per_environment"],
            "early_resets_per_environment_corrected": corrected["early_resets_per_environment"],
            "right_ankle_contact_frame_fraction_raw": raw["right_ankle_contact_frame_fraction"],
            "right_ankle_contact_frame_fraction_corrected": corrected["right_ankle_contact_frame_fraction"],
            "kick_event_proxy_retained_count": semantic["event_amplitude_and_peak_retained_count"],
            "mean_peak_added_abs_velocity_rad_s": temporal["mean_peak_added_abs_velocity_rad_s"],
        })

    frozen = {}
    baseline_ids = None
    frozen_goal = None
    for arm in ("raw", "cap025", "cap050", "cap100"):
        item = load(f"r3_i1_frozen_goal_seed42_{arm}.json")
        if (item["num_envs"], item["video_length"], item["seed"], item["motion_library_size"]) != (256, 500, 42, 64):
            raise ValueError(f"Unexpected frozen-goal setup {arm}")
        if baseline_ids is None:
            baseline_ids = item["initial_motion_ids"]
            frozen_goal = item["frozen_task_goal_from_dir"]
        elif item["initial_motion_ids"] != baseline_ids or item["frozen_task_goal_from_dir"] != frozen_goal:
            raise ValueError(f"Mismatched frozen-goal initial conditions {arm}")
        frozen[arm] = {
            key: item[key] for key in (
                "official_goal_proximity_success_percent", "joint_mae_rad",
                "early_resets_per_environment", "left_ankle_contact_frame_fraction",
                "right_ankle_contact_frame_fraction",
            )
        }
    efficiency = load("r3_i1_efficiency_audit.json")
    if [row["seed"] for row in efficiency["rows"]] != [42, 43, 44]:
        raise ValueError("Efficiency seeds differ")
    output = {
        "status": "complete_diagnostics_i1_gate_not_passed",
        "evaluation_types": ["simulation_only_fixed_tracker", "semantic_kinematic_proxy", "wall_time_component_timing"],
        "generated_per_seed": 100,
        "paired_seed_rows": paired,
        "frozen_raw_goal_seed42": frozen,
        "efficiency_rows": efficiency["rows"],
        "external_pre_generation_goal_frozen_in_simulation": False,
        "target_impact_contact_measured": False,
        "multi_text_semantic_benchmark_measured": False,
        "motor_torque_or_hardware_feasibility_measured": False,
        "same_budget_full_i1_ablation_completed": False,
        "input_result_sha256": sources,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(paired)} generation seeds and {len(frozen)} frozen-goal arms")


if __name__ == "__main__":
    main()
