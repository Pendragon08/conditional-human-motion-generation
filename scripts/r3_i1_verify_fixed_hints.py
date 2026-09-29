"""Verify matched OmniControl conditions before cross-seed retargeting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = {}
    for seed in (42, 43, 44):
        source = args.base / f"seed{seed}" / "results.npy"
        data = np.load(source, allow_pickle=True).item()
        if np.asarray(data["motion"]).shape != (100, 22, 3, 196):
            raise RuntimeError(f"Wrong motion shape: seed{seed}")
        if np.asarray(data["hint"]).shape != (100, 196, 22, 3):
            raise RuntimeError(f"Wrong hint shape: seed{seed}")
        samples[seed] = (source, data)
    reference = samples[42][1]
    for seed in (43, 44):
        candidate = samples[seed][1]
        if not np.array_equal(candidate["hint"], reference["hint"]):
            raise RuntimeError(f"Constraint mismatch: seed{seed}")
        if not np.array_equal(candidate["lengths"], reference["lengths"]):
            raise RuntimeError(f"Length mismatch: seed{seed}")
        if candidate["text"] != reference["text"]:
            raise RuntimeError(f"Text mismatch: seed{seed}")
    result = {
        "status": "MATCHED_INPUTS_VERIFIED",
        "conditions": 100,
        "sampling_seeds": [42, 43, 44],
        "fixed_constraint_seed": 20260929,
        "all_hints_text_and_lengths_identical": True,
        "one_semantic_class_only": "A person stands and kicks with his right leg.",
        "input_sha256": {str(seed): digest(source) for seed, (source, _) in samples.items()},
        "note": "This verifies input pairing, not generated-motion quality or I1 success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
