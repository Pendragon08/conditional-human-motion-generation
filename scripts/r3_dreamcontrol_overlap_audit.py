"""Check whether regenerated DreamControl references duplicate supplied ones."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load(folder):
    result = {}
    for path in folder.glob("*.pkl"):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        result[path.name] = {
            "joints": np.asarray(data["joints"]),
            "root": np.asarray(data["global_position"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplied", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supplied = load(args.supplied)
    regenerated = load(args.regenerated)
    shared = sorted(set(supplied) & set(regenerated))
    rows = []
    for name in shared:
        a, b = supplied[name], regenerated[name]
        rows.append({
            "file": name,
            "joint_rmse_rad": float(np.sqrt(np.mean((a["joints"] - b["joints"]) ** 2))),
            "root_rmse_m": float(np.sqrt(np.mean((a["root"] - b["root"]) ** 2))),
            "joint_exact": bool(np.array_equal(a["joints"], b["joints"])),
            "root_exact": bool(np.array_equal(a["root"], b["root"])),
        })
    result = {
        "supplied_count": len(supplied),
        "regenerated_count": len(regenerated),
        "shared_filename_count": len(shared),
        "exact_joint_duplicates_by_name": sum(row["joint_exact"] for row in rows),
        "exact_root_duplicates_by_name": sum(row["root_exact"] for row in rows),
        "matching_name_joint_rmse_median_rad": float(np.median([row["joint_rmse_rad"] for row in rows])) if rows else None,
        "matching_name_root_rmse_median_m": float(np.median([row["root_rmse_m"] for row in rows])) if rows else None,
        "new_filenames": sorted(set(regenerated) - set(supplied)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
