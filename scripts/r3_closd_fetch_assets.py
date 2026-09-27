"""Stage the author's CLoSD assets after confirming their separate licenses.

Run with HF_HOME on /data and a private HTTP proxy when this server cannot
reach Hugging Face directly. Do not copy assets into Git or the local PC.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO = "guytevet/CLoSD"
EXPECTED_REVISION = "de7106b947b6f70700b5320d1cd61fef4a9ebc9b"
REQUIRED = [
    "checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth",
    "checkpoints/closd/CLoSD_no_finetune/Humanoid.pth",
    "checkpoints/dip/DiP_multi-target_10steps_context20_predict40/model000300000.pt",
    "data/amass/amass_train_take6_upright.pkl",
    "data/smpl/SMPL_NEUTRAL.pkl",
    "data/dataset/HumanML3D/Mean.npy",
    "data/dataset/HumanML3D/Std.npy",
]


def main() -> None:
    home = Path(os.environ["HF_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    print(f"CLOSD_ASSET_REPO={REPO}", flush=True)
    print(f"CLOSD_ASSET_EXPECTED_REVISION={EXPECTED_REVISION}", flush=True)
    path = Path(snapshot_download(repo_id=REPO, revision="main", max_workers=2))
    if path.name != EXPECTED_REVISION:
        raise RuntimeError(f"Unexpected Hub revision: {path.name}")
    missing = [name for name in REQUIRED if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty required assets: {missing}")
    files = [p for p in path.rglob("*") if p.is_file()]
    manifest = {
        "repo": REPO,
        "revision": path.name,
        "path": str(path),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "required": {name: (path / name).stat().st_size for name in REQUIRED},
    }
    output = Path(os.environ["CLOSD_ASSET_MANIFEST"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("CLOSD_ASSET_COMPLETE=" + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
