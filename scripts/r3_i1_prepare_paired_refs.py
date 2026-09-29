"""Create equal-ID, read-only reference views for a paired robot evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-goals", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_goals.exists():
        raise FileExistsError("Paired output already exists")
    base = {path.name: path for path in args.baseline.glob("*.pkl")}
    candidate = {path.name: path for path in args.candidate.glob("*.pkl")}
    paired = sorted(set(base) & set(candidate), key=lambda name: int(Path(name).stem))
    if not paired:
        raise ValueError("No matching reference IDs")
    goal_data = json.loads(args.goals.read_text())
    by_name = {f'{row["sample"]}.pkl': row for row in goal_data["rows"]}
    if not set(paired) <= set(by_name):
        raise ValueError("Missing pre-generation goals for paired IDs")
    for arm, source in (("baseline", base), ("candidate", candidate)):
        directory = args.output_dir / arm
        directory.mkdir(parents=True)
        for name in paired:
            (directory / name).symlink_to(source[name].resolve())
    output = {
        **goal_data,
        "status": "paired_fixed_goal_no_extra_ik",
        "rows": [by_name[name] for name in paired],
        "paired_motion_ids": [int(Path(name).stem) for name in paired],
        "baseline_sha256": {name: digest(base[name]) for name in paired},
        "candidate_sha256": {name: digest(candidate[name]) for name in paired},
    }
    args.output_goals.parent.mkdir(parents=True, exist_ok=True)
    args.output_goals.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
