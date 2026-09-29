"""Summarize 100-input paired generation and unchanged-retarget audits.

No rejected generated motion is dropped from the end-to-end yield denominator.
The same-ID subset is used only for paired differences; it is labelled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = {"protocol": "no_extra_ik_generation_then_original_retarget", "seeds": {}}
    for seed in (42, 43, 44):
        paths = {
            "human": args.results / f"r3_i1_joint_v2_human_seed{seed}.json",
            "baseline": args.results / f"r3_i1_fixed_seed{seed}_audit.json",
            "candidate": args.results / f"r3_i1_joint_v2_robot_seed{seed}.json",
        }
        human, base, cand = (json.loads(paths[k].read_text()) for k in paths)
        if human["sample_count"] != 100 or len(base["rows"]) != 100 or len(cand["rows"]) != 100:
            raise ValueError(f"Seed {seed} missing generated inputs")
        base_constraint_sha = base.get("constraint_source_sha256", base["source_npy_sha256"])
        if base_constraint_sha != cand["constraint_source_sha256"]:
            raise ValueError(f"Seed {seed} uses different original hints")
        if base["retarget_source_sha256"] != cand["retarget_source_sha256"]:
            raise ValueError(f"Seed {seed} uses different retargeter code")
        b = {int(row["sample"]): row for row in base["rows"]}
        c = {int(row["sample"]): row for row in cand["rows"]}
        if set(b) != set(c) or set(b) != set(range(100)):
            raise ValueError(f"Seed {seed} has mismatched input IDs")
        both = [idx for idx in range(100) if b[idx]["retargeted"] and c[idx]["retargeted"]]
        if not both:
            raise ValueError(f"Seed {seed} has no paired retargets")
        source = {arm: human["summary"][arm] for arm in ("baseline", "joint_v2")}
        summary["seeds"][str(seed)] = {
            "source_sha256": {k: sha(v) for k, v in paths.items()},
            "all_generated_inputs": 100,
            "baseline_retargeted": sum(bool(row["retargeted"]) for row in b.values()),
            "candidate_retargeted": sum(bool(row["retargeted"]) for row in c.values()),
            "same_id_retarget_count": len(both),
            "same_id_retarget_ids": both,
            "all_input_human_metrics": source,
            "same_id_robot_mean_error_m": {
                "baseline": mean(b[idx]["robot_mean_error_m"] for idx in both),
                "joint_v2": mean(c[idx]["robot_mean_error_m"] for idx in both),
            },
            "same_id_robot_success_at_15cm": {
                "baseline": mean(b[idx]["robot_mean_error_m"] <= 0.15 for idx in both),
                "joint_v2": mean(c[idx]["robot_mean_error_m"] <= 0.15 for idx in both),
            },
            "full_input_robot_success_at_15cm": {
                "baseline": sum(row["retargeted"] and row["robot_mean_error_m"] <= 0.15
                                for row in b.values()) / 100,
                "joint_v2": sum(row["retargeted"] and row["robot_mean_error_m"] <= 0.15
                                for row in c.values()) / 100,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
