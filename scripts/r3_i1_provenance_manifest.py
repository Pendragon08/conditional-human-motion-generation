"""Record hashes and completion receipts for I1 runs without publishing assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SEEDS = (42, 43, 44)
STATE_NAMES = (
    "r3_i1_fixed_candidates_gpu1.state",
    "r3_i1_fixed_retarget_queue.state",
    "r3_i1_fixed_eval_queue.state",
    "r3_i1_export_corrected_refs.state",
    "r3_i1_semantic_contact_queue.state",
    "r3_i1_physics_paired_queue.state",
    "r3_i1_cap_sweep_seed42.state",
    "r3_i1_cap_physics_queue.state",
    "r3_i1_frozen_goal_physics_queue.state",
)
SCRIPT_NAMES = (
    "r3_i1_robotspace_ik_pilot.py",
    "r3_i1_semantic_contact_proxy.py",
    "r3_i1_efficiency_audit.py",
    "r3_i1_play_eval_metrics.py",
    "r3_i1_temporal_quality.py",
    "r3_i1_pkl_field_integrity.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    monitor = root / "artifacts/monitoring"
    stages = {seed: root / f"third_party/DreamControl/TrajGen/sample_i1_fixed_seed{seed}" for seed in SEEDS}
    assets = {}
    for seed, stage in stages.items():
        assets[str(seed)] = {
            "condition_results_npy": record(stage / "Kick_sim/results.npy"),
            "retarget_source": record(stage / "Kick_sim/retarget.py"),
            "raw_reference_count": len(list((stage / "Kick_sim1").glob("*.pkl"))),
            "corrected_reference_count": len(list((root / f"artifacts/i1_corrected_seed{seed}").glob("*.pkl"))),
        }
    urdf = root / "third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    states = {}
    for name in STATE_NAMES:
        path = monitor / name
        if not path.exists():
            raise FileNotFoundError(path)
        states[name] = {**record(path), "content": path.read_text(errors="replace").strip()}
    logs = {path.name: record(path) for path in sorted(monitor.glob("r3_i1_*.log"))}
    scripts = {name: record(root / "scripts" / name) for name in SCRIPT_NAMES}
    upstream = root / "third_party/DreamControl"
    try:
        upstream_top = Path(subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "--show-toplevel"], text=True
        ).strip()).resolve()
        upstream_commit = (subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip() if upstream_top == upstream.resolve() else None)
    except (OSError, subprocess.CalledProcessError):
        upstream_commit = None
    result = {
        "status": "hash_receipt_not_full_asset_archive",
        "root": str(root),
        "upstream_dreamcontrol_checkout_commit_verified": upstream_commit,
        "upstream_version_note": "Null means this bundled source directory is not itself a Git checkout; source-file hashes are the concrete revision evidence.",
        "assets": assets,
        "urdf": record(urdf),
        "policy_checkpoint": record(args.checkpoint),
        "states": states,
        "log_receipts": logs,
        "script_receipts": scripts,
        "note": "Raw motions, generated PKL files and full logs remain on the server; this manifest does not grant redistribution rights or independently reconstruct the experiments.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {len(states)} states, {len(logs)} logs, {len(scripts)} scripts")


if __name__ == "__main__":
    main()
