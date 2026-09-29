"""Audit measured OmniControl sampling and added IK wall time on the same host.

The official progress bar reports denoising-loop time. It does not include all
model loading, retargeting, or postprocessing and is not a full pipeline time.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    rows = []
    for seed in (42, 43, 44):
        sampling_log = root / f"artifacts/monitoring/r3_i1_fixed_candidates_seed{seed}.log"
        log = sampling_log.read_text(errors="replace")
        matches = re.findall(r"1000/1000 \[([0-9:]+)<00:00", log)
        if not matches:
            raise ValueError(f"No completed denoising progress bar: {sampling_log}")
        batch_seconds = seconds(matches[-1])
        exports = json.loads((root / f"results/r3_i1_corrected_export_seed{seed}.json").read_text())
        times = [float(row["optimization_wall_time_s"]) for row in exports["rows"]]
        if len(times) not in (63, 64):
            raise ValueError(f"Unexpected retained count: {seed} {len(times)}")
        rows.append({
            "seed": seed,
            "generated": 100,
            "retargeted_and_ik_corrected": len(times),
            "official_denoising_steps": 1000,
            "sampler_progress_bar_seconds_for_100": batch_seconds,
            "sampler_amortized_seconds_per_generated_motion": batch_seconds / 100,
            "added_ik_adam_steps_per_retained_motion": exports["steps"],
            "added_ik_wall_seconds_mean_per_retained_motion": statistics.mean(times),
            "added_ik_wall_seconds_median_per_retained_motion": statistics.median(times),
            "added_ik_wall_seconds_p90_per_retained_motion": sorted(times)[int(0.9 * (len(times) - 1))],
            "added_ik_wall_seconds_total": sum(times),
        })
    result = {
        "status": "measured_additional_cost_not_efficiency_gain",
        "hardware": "server GPU 1 NVIDIA RTX A6000 for diffusion; CPU for constrained IK",
        "same_100_fixed_hints_per_seed": True,
        "rows": rows,
        "limitations": [
            "Sampler duration is tqdm denoising-loop time, not full application wall time.",
            "IK timer includes per-motion file loading and forward kinematics; it was measured sequentially on CPU.",
            "Retargeting, model load, and simulator inference are outside this cost comparison.",
            "The current correction adds 80 optimization steps per retained motion; no inference-efficiency improvement is established.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
