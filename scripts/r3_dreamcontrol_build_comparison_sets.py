"""Build immutable DreamControl reference sets for the seed42 data-utility test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "third_party/DreamControl/TrajGen"
BASE = TRAJ / "sample/Kick_sim2"
NEW = TRAJ / "sample_r3_n100_seed42/Kick_sim2"
OUT = TRAJ / "r3_comparison_seed42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_exact(directory: Path, name: str, source: Path) -> None:
    target = directory / name
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to replace {target}")
    target.symlink_to(source.resolve())


def main() -> None:
    base = sorted(BASE.glob("*.pkl"))
    generated = sorted(NEW.glob("*.pkl"))
    if len(base) != 66 or len(generated) != 64:
        raise RuntimeError(f"Expected 66 base and 64 generated: {len(base)}, {len(generated)}")

    # Stable 32/32 split independent of filesystem enumeration order.
    ranked = sorted(generated, key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    augmentation, heldout = ranked[:32], ranked[32:]
    paths = {
        "base_train": OUT / "base_train",
        "augmented_train": OUT / "augmented_train",
        "generated_holdout": OUT / "generated_holdout",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            # Existing dataset is accepted only when each entry is the expected link.
            expected = {
                "base_train": 66,
                "augmented_train": 98,
                "generated_holdout": 32,
            }[directory.name]
            if len(list(directory.iterdir())) != expected:
                raise RuntimeError(f"Unexpected pre-existing content: {directory}")

    for source in base:
        link_exact(paths["base_train"], "base_" + source.name, source)
        link_exact(paths["augmented_train"], "base_" + source.name, source)
    for source in augmentation:
        link_exact(paths["augmented_train"], "gen42_" + source.name, source)
    for source in heldout:
        link_exact(paths["generated_holdout"], "gen42_" + source.name, source)

    assert len(list(paths["base_train"].glob("*.pkl"))) == 66
    assert len(list(paths["augmented_train"].glob("*.pkl"))) == 98
    assert len(list(paths["generated_holdout"].glob("*.pkl"))) == 32
    manifest = {
        "purpose": "fixed-budget comparison; old policy trains on base66, new policy on base66+seed42/32; both evaluate on held-out seed42/32",
        "split_rule": "SHA256(filename), first 32 generated to augmentation, remainder to holdout",
        "base": [{"filename": p.name, "sha256": sha256(p)} for p in base],
        "augmentation": [{"filename": p.name, "sha256": sha256(p)} for p in augmentation],
        "holdout": [{"filename": p.name, "sha256": sha256(p)} for p in heldout],
        "directories": {key: str(path) for key, path in paths.items()},
    }
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Existing manifest differs: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"base": 66, "augmented_train": 98, "generated_holdout": 32, "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
