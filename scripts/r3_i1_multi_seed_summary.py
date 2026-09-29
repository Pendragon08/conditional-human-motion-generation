"""Aggregate fixed-condition, three-seed G1 constraint correction diagnostics."""

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "r3_i1_fixed_hints_manifest.json").read_text())
    if manifest["status"] != "MATCHED_INPUTS_VERIFIED":
        raise RuntimeError("Matched constraints have not been verified")
    rows = []
    for seed in (42, 43, 44):
        correction = json.loads((args.results / f"r3_i1_fixed_seed{seed}_ik.json").read_text())
        audit = json.loads((args.results / f"r3_i1_fixed_seed{seed}_audit.json").read_text())
        data = correction["rows"]
        if len(data) != audit["paired_robot"]["count"]:
            raise RuntimeError(f"Retarget counts differ for seed{seed}")
        success = [r["after_error_m"] <= 0.15 for r in data]
        smooth = [r["max_speed_change_rad_per_frame"] <= 0.1 for r in data]
        rows.append({
            "seed": seed,
            "generated": 100,
            "retargeted": len(data),
            "not_retargeted": 100 - len(data),
            "retargeted_mean_before_m": mean(r["before_error_m"] for r in data),
            "retargeted_mean_after_m": mean(r["after_error_m"] for r in data),
            "retargeted_improved_count": sum(r["after_error_m"] < r["before_error_m"] for r in data),
            "retargeted_success_15cm_count": sum(success),
            "retargeted_success_and_speed_screen_count": sum(a and b for a, b in zip(success, smooth)),
            "all_generated_success_15cm_rate": sum(success) / 100,
            "all_generated_success_and_speed_screen_rate": sum(a and b for a, b in zip(success, smooth)) / 100,
            "joint_limit_violations": sum(r["joint_limit_violations"] for r in data),
        })
    result = {
        "status": "three_seed_component_result_not_full_I1_Gate",
        "same_100_constraints_verified": True,
        "seeds": rows,
        "limits": [
            "one Kick text class; no semantic generalization",
            "rejected retargets counted as failure in all-generated rates",
            "0.1 rad/frame speed screen is heuristic, not dynamic or safety validation",
            "post-retarget IK component, not improved diffusion generation",
            "no downstream physics rollout, contact test, or efficiency comparison",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
