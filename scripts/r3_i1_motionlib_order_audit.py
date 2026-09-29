"""Check whether official unsorted glob order maps motion IDs to same clips."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    rows = []
    for seed in (42, 43, 44):
        stage = a.root / f"third_party/DreamControl/TrajGen/sample_i1_fixed_seed{seed}/Kick_sim1"
        corrected = a.root / f"artifacts/i1_corrected_seed{seed}"
        raw_order = [Path(path).name for path in glob.glob(str(stage / "*.pkl"))]
        corrected_order = [Path(path).name for path in glob.glob(str(corrected / "*.pkl"))]
        rows.append({"seed": seed, "same_glob_order": raw_order == corrected_order,
                     "same_name_set": set(raw_order) == set(corrected_order),
                     "raw_order": raw_order, "corrected_order": corrected_order})
    result = {"status": "official_motion_lib_glob_order_audit",
              "all_same_glob_order": all(r["same_glob_order"] for r in rows),
              "all_same_name_set": all(r["same_name_set"] for r in rows),
              "rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
