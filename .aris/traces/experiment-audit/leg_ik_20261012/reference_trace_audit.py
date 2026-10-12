"""Read-only adjudication of replay-motion and applied-reference trace bindings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def motion_hash(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for key in sorted(arrays.files):
            value = np.ascontiguousarray(arrays[key])
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def max_abs(left, right):
    return float(np.max(np.abs(left - right)))


def main(root, output):
    out = root / "results/support_leg_ik_20261012"
    parent = root / "results/support_reference_probe_v63_20261012"
    source = root / "results/paired_feedback_validation_v51_20261012"
    reports = {}
    for arm, home in (("baseline", parent), ("cartesian", out), ("two_link_ik", out)):
        reference_path = out / "references" / (arm + ".npz")
        request_path = home / "requests" / ("val_" + arm + "_s11.json")
        raw = home / "raw" / ("val_" + arm + "_s11")
        request = read(request_path)
        metadata = read(raw / "metadata.json")
        lineage = read(raw / "lineage.json")
        with np.load(reference_path, allow_pickle=False) as archive:
            selected = archive["reference"]
        canonical = (selected - selected[:, :, 0:1, :]) + selected[:, :, 0:1, :]
        with np.load(raw / "trajectories.npz", allow_pickle=False) as archive:
            actual = archive["reference"]
            valid = archive["valid"]
        with np.load(raw / "observation_trace.npz", allow_pickle=False) as archive:
            trace = {name: archive[name] for name in archive.files}
        expected_velocity = (trace["ref_position"] - trace["ref_previous"]) / metadata["dt"]
        checks = {
            "request_motion_hash_uses_canonical_npz_content": request["replay_motion"] == motion_hash(reference_path),
            "lineage_request_exact": lineage["request"] == request,
            "raw_reference_equals_upstream_canonical_transform": np.array_equal(actual, canonical),
            "trace_position_equals_applied_reference": np.array_equal(trace["ref_position"], actual),
            "trace_first_previous_equals_current": np.array_equal(trace["ref_previous"][0], trace["ref_position"][0]),
            "trace_subsequent_previous_equals_prior_position": np.array_equal(trace["ref_previous"][1:], trace["ref_position"][:-1]),
            "trace_velocity_matches_position_difference": np.array_equal(trace["ref_velocity"], expected_velocity),
            "trace_first_velocity_zero": not np.any(trace["ref_velocity"][0]),
            "all_reference_frames_valid": bool(valid.all()),
            "metadata_rule_explicit": metadata.get("reference_velocity_rule") == "Successive applied canonical positions / s_dt; first frame zero; repeated query preserves previous; diagnostic amendment, not innovation",
        }
        reports[arm] = {
            "checks": checks,
            "all_pass": all(checks.values()),
            "request_replay_motion": request["replay_motion"],
            "recomputed_motion_hash": motion_hash(reference_path),
            "selected_to_canonical_max_abs_m": max_abs(selected, canonical),
            "raw_to_selected_max_abs_m": max_abs(actual, selected),
            "raw_to_canonical_max_abs_m": max_abs(actual, canonical),
            "trace_velocity_max_abs_error_m_s": max_abs(trace["ref_velocity"], expected_velocity),
        }
    source_request = read(source / "requests/val_shared_s11_r0.json")
    source_lineage = read(source / "raw/val_shared_s11_r0/lineage.json")
    source_checks = {
        "mode_is_native_construction": source_request["mode"] == "native",
        "scope_is_val_seed11_round0": all(
            source_request[key] == value
            for key, value in {"split": "val", "seed": 11, "round": 0, "noise_round": 0}.items()
        ),
        "lineage_request_exact": source_lineage["request"] == source_request,
        "execution_bound": source_lineage["certification"] == "bound_request_at_execution",
    }
    result = {
        "status": "PASS" if all(row["all_pass"] for row in reports.values()) and all(source_checks.values()) else "FAIL",
        "purpose": "Adjudicate two non-contract false flags retained by independent_recompute.json and verify the applied-reference velocity trace without rerunning any simulator.",
        "reports": reports,
        "source_construction_checks": source_checks,
        "simulation_calls": 0,
        "generation_calls": 0,
        "training_calls": 0,
        "limitations": "This validates saved-array and request construction semantics only; it is not efficacy evidence.",
    }
    with output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
    print("REFERENCE_TRACE_AUDIT_" + result["status"])
    print(json.dumps({arm: report["checks"] for arm, report in reports.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output)
