"""Freeze an equal-ID no-extra-IK reference set for the original Kick task.

The script never edits a trajectory. All rejected generated inputs remain in
the denominator of the separate end-to-end generation-yield report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retarget-script", type=Path, required=True)
    parser.add_argument("--baseline-generation", type=Path, required=True)
    parser.add_argument("--candidate-generation", type=Path, required=True)
    parser.add_argument("--generated-count", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists() or args.manifest.exists():
        raise FileExistsError("Paired experiment is already frozen")
    base = {int(p.stem): p for p in args.baseline.glob("*.pkl")}
    candidate = {int(p.stem): p for p in args.candidate.glob("*.pkl")}
    paired = sorted(set(base) & set(candidate))
    if not paired:
        raise ValueError("No common reference IDs")
    if any(idx < 0 or idx >= args.generated_count for idx in set(base) | set(candidate)):
        raise ValueError("Retarget output ID outside generated input range")
    for arm, source in (("baseline", base), ("candidate", candidate)):
        folder = args.output_dir / arm
        folder.mkdir(parents=True)
        for idx in paired:
            (folder / f"{idx}.pkl").symlink_to(source[idx].resolve())
    manifest = {
        "protocol": "original_kick_task_equal_ids_no_extra_ik",
        "generated_input_count_per_arm": args.generated_count,
        "baseline_retarget_count": len(base),
        "candidate_retarget_count": len(candidate),
        "paired_train_count": len(paired),
        "paired_train_ids": paired,
        "unpaired_baseline_ids": sorted(set(base) - set(candidate)),
        "unpaired_candidate_ids": sorted(set(candidate) - set(base)),
        "same_official_task_and_control_budget_required": True,
        "retarget_script_sha256": sha256(args.retarget_script),
        "generation_sha256": {
            "baseline": sha256(args.baseline_generation),
            "candidate": sha256(args.candidate_generation),
        },
        "retargeted_pkl_sha256": {
            "baseline": {str(idx): sha256(base[idx]) for idx in paired},
            "candidate": {str(idx): sha256(candidate[idx]) for idx in paired},
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
