"""Compare newly sampled human motion against the seed-10 reproduction."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = np.load(args.baseline, allow_pickle=True).item()["motion"]
    b = np.load(args.candidate, allow_pickle=True).item()["motion"]
    assert a.shape == b.shape == (100, 22, 3, 196)
    assert np.isfinite(b).all()
    rmse = np.sqrt(np.mean((a - b) ** 2, axis=(1, 2, 3)))
    summary = {
        "count": 100,
        "shape": list(b.shape),
        "paired_human_motion_rmse_median_m": float(np.median(rmse)),
        "paired_human_motion_rmse_min_m": float(rmse.min()),
        "paired_human_motion_rmse_max_m": float(rmse.max()),
        "paired_clips_over_1cm": int((rmse > 0.01).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    assert summary["paired_human_motion_rmse_median_m"] > 0.01


if __name__ == "__main__":
    main()
