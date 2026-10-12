"""Independent read-only integrity recomputation for the leg-IK diagnosis.

This audit intentionally does not import the frozen study analyzer or the shared
metric module.  It reconstructs the published metrics directly from saved
arrays and writes only the explicitly requested audit evidence file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = (
    "endpoint_error_m",
    "heading_error_rad",
    "endpoint_time_error_s",
    "surface_slip_proxy_fraction",
    "penetration_fraction",
    "jerk_p95_m_s3",
    "tracking_mean_m",
    "tracking_p95_m",
)
COUNT_FIELDS = (
    "completed",
    "condition_pass",
    "physical_pass",
    "tracking_pass",
    "joint_pass",
)
INITIAL_FIELDS = {
    "_humanoid_root_states",
    "_dof_pos",
    "_dof_vel",
    "_rigid_body_pos",
    "_rigid_body_rot",
    "_rigid_body_vel",
    "_rigid_body_ang_vel",
}
CACHE_FIELDS = {
    "_rigid_body_pos",
    "_rigid_body_rot",
    "_rigid_body_vel",
    "_rigid_body_ang_vel",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def heading(positions, names):
    across = (
        positions[..., names.index("R_Hip"), :]
        - positions[..., names.index("L_Hip"), :]
        + positions[..., names.index("R_Shoulder"), :]
        - positions[..., names.index("L_Shoulder"), :]
    )
    forward = np.cross(np.array([0.0, 0.0, 1.0]), across)
    return np.arctan2(forward[..., 1], forward[..., 0])


def rotate_xyzw(q, v):
    u, w = q[..., :3], q[..., 3:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def surface_metrics(pos, quat, linvel, angvel, force, names, geometry, thresholds):
    speeds = []
    support = []
    heights = []
    for name, geom in geometry.items():
        index = names.index(name)
        points = np.asarray(geom["points"], dtype=float)
        radii = np.asarray(geom["radii"], dtype=float)
        rotated = rotate_xyzw(quat[:, index, None, :], points[None])
        world = pos[:, index, None, :] + rotated
        world[:, :, 2] -= radii[None]
        lowest = world[np.arange(len(pos)), np.argmin(world[:, :, 2], axis=1)]
        lever = lowest - pos[:, index]
        com = rotate_xyzw(quat[:, index], np.asarray(geom["com_local"], dtype=float))
        velocity = linvel[:, index] + np.cross(angvel[:, index], lever - com)
        supported = (
            (force[:, index, 2] > thresholds["contact_force_n"])
            & (lowest[:, 2] < thresholds["surface_near_ground_m"])
        )
        speeds.append(np.linalg.norm(velocity[:, :2], axis=1))
        support.append(supported)
        heights.append(lowest[:, 2])
    speeds = np.stack(speeds, axis=1)
    support = np.stack(support, axis=1)
    heights = np.stack(heights, axis=1)
    count = int(support.sum())
    return {
        "surface_slip_proxy_mean_m_s": float(speeds[support].mean()) if count else None,
        "surface_slip_proxy_fraction": float(
            (speeds[support] > thresholds["sliding_speed_m_s"]).mean()
        )
        if count
        else None,
        "support_observations": count,
        "penetration_fraction": float(
            (heights < -thresholds["penetration_depth_m"]).any(axis=1).mean()
        ),
        "penetration_depth_p95_m": float(np.quantile(np.maximum(-heights, 0), 0.95)),
        "airborne_fraction": float((~support.any(axis=1)).mean()),
        "contact_metric_type": "COM velocity corrected body-force and collision-primitive surface proxy",
    }


def recompute_row(arrays, index, metadata, thresholds):
    valid = arrays["valid"][:, index]
    length = int(valid.sum())
    expected = int(metadata["episode_steps"])
    names = metadata["names"]
    dt = metadata["dt"]
    target = np.asarray(metadata["original_goal"][index])
    joint = names.index(metadata["cases"][index]["goal_joint_body"])
    goal_heading = float(metadata["original_heading"][index])
    row = {
        key: metadata["cases"][index][key]
        for key in ("case_id", "source_id", "clip_id", "family", "offset_m")
    }
    row.update(
        length=length,
        terminated=bool(arrays["terminated"][index]),
        completed=length == expected and not bool(arrays["terminated"][index]),
        original_goal=target.tolist(),
        original_heading=goal_heading,
        seed=metadata["seed"],
        mode=metadata["mode"],
        round=metadata["round"],
    )
    if not length:
        row.update(joint_pass=False, status="no_valid_steps")
        return row
    pos = arrays["actual"][valid, index]
    reference = arrays["reference"][valid, index]
    distances = np.linalg.norm(pos[:, joint] - target, axis=-1)
    row["endpoint_error_m"] = float(distances[-1])
    row["endpoint_error_vector"] = (target - pos[-1, joint]).tolist()
    row["heading_error_rad"] = float(abs(wrap_angle(heading(pos[-1], names) - goal_heading)))
    row["heading_error_signed_rad"] = float(wrap_angle(goal_heading - heading(pos[-1], names)))
    prediction_start = metadata["context_steps"]
    if length > prediction_start:
        distances_after_prefix = distances[prediction_start:]
        near = (
            np.flatnonzero(distances_after_prefix <= distances_after_prefix.min() + 0.001)
            + prediction_start
        )
        best = near[np.argmin(np.abs(near - (expected - 1)))]
        row["endpoint_time_error_s"] = float(abs(best - (expected - 1)) * dt)
    else:
        row["endpoint_time_error_s"] = expected * dt
    start = min(prediction_start, length - 1)
    errors = np.linalg.norm(pos[start:] - reference[start:], axis=-1)
    row["tracking_mean_m"] = float(errors.mean())
    row["tracking_p95_m"] = float(np.quantile(errors.mean(axis=1), 0.95))
    velocity = np.diff(pos[start:], axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    row["jerk_p95_m_s3"] = (
        float(np.quantile(np.linalg.norm(jerk, axis=-1), 0.95)) if len(jerk) else None
    )
    geometry = {
        name: dict(
            geom,
            com_local=metadata["native_body_properties"][index][names.index(name)]["com"],
        )
        for name, geom in metadata["foot_geometry"].items()
    }
    row.update(
        surface_metrics(
            pos[start:],
            arrays["quaternion"][valid, index][start:],
            arrays["linvel"][valid, index][start:],
            arrays["angvel"][valid, index][start:],
            arrays["contact_force"][valid, index][start:],
            names,
            geometry,
            thresholds,
        )
    )
    condition = (
        row["endpoint_error_m"] <= thresholds["goal_position_m"]
        and row["heading_error_rad"] <= thresholds["heading_rad"]
        and row["endpoint_time_error_s"] <= thresholds["goal_time_s"] + 1e-6
    )
    physical = (
        row["penetration_fraction"] <= thresholds["penetration_fraction"]
        and row["surface_slip_proxy_fraction"] is not None
        and row["surface_slip_proxy_fraction"] <= thresholds["sliding_fraction"]
        and row["jerk_p95_m_s3"] is not None
        and row["jerk_p95_m_s3"] <= thresholds["continuity_jerk_p95_m_s3"]
    )
    tracking = (
        row["tracking_mean_m"] <= thresholds["tracking_mean_m"]
        and row["tracking_p95_m"] <= thresholds["tracking_p95_m"]
    )
    row.update(
        condition_pass=bool(condition),
        physical_pass=bool(physical),
        tracking_pass=bool(tracking),
        joint_pass=bool(row["completed"] and condition and physical and tracking),
        status="measured",
        semantic_status="not_verified_by_independent_evaluator",
    )
    return row


def compare_value(left, right, path, mismatches):
    if left is None or right is None:
        if left is not right:
            mismatches.append({"path": path, "stored": left, "recomputed": right})
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            mismatches.append({"path": path, "stored": left, "recomputed": right})
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            if not (math.isnan(float(left)) and math.isnan(float(right))):
                mismatches.append({"path": path, "stored": left, "recomputed": right})
        elif not np.isclose(float(left), float(right), atol=1e-10, rtol=1e-10):
            mismatches.append({"path": path, "stored": left, "recomputed": right})
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            mismatches.append({"path": path + ".length", "stored": len(left), "recomputed": len(right)})
            return
        for index, (a, b) in enumerate(zip(left, right)):
            compare_value(a, b, f"{path}[{index}]", mismatches)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            mismatches.append(
                {
                    "path": path + ".keys",
                    "stored_only": sorted(set(left) - set(right)),
                    "recomputed_only": sorted(set(right) - set(left)),
                }
            )
        for key in sorted(set(left) & set(right)):
            compare_value(left[key], right[key], f"{path}.{key}", mismatches)
        return
    if left != right:
        mismatches.append({"path": path, "stored": left, "recomputed": right})


def verify_batch(folder: Path, request_path: Path, reference_path: Path, versions_path: Path, manifest_path: Path, thresholds_path: Path):
    request = read_json(request_path)
    lineage = read_json(folder / "lineage.json")
    receipt = read_json(folder / "receipt.json")
    required = {"metadata.json", "metrics.json", "trajectories.npz"}
    checks = {
        "lineage_execution_bound": lineage.get("certification") == "bound_request_at_execution",
        "lineage_request_exact": lineage.get("request") == request,
        "lineage_output_names_exact": set(lineage.get("outputs", {})) == required,
        "receipt_complete": receipt.get("status") == "complete",
        "request_reference_hash": request.get("replay_motion") == sha256(reference_path),
        "request_versions_hash": request.get("versions") == sha256(versions_path),
        "request_manifest_hash": request.get("manifest") == sha256(manifest_path),
        "request_thresholds_hash": request.get("thresholds") == sha256(thresholds_path),
        "request_seed_11": request.get("seed") == 11,
        "request_scope": all(
            request.get(key) == value
            for key, value in {
                "split": "val",
                "mode": "replay",
                "round": 0,
                "noise_round": 0,
            }.items()
        ),
    }
    for name in required:
        checks[f"lineage_hash_{name}"] = lineage.get("outputs", {}).get(name) == sha256(folder / name)
    checks["receipt_trajectory_hash"] = (
        receipt.get("trajectories_sha256") == lineage.get("outputs", {}).get("trajectories.npz")
    )
    log_path = folder.parents[1] / (folder.name + ".log")
    log_text = log_path.read_text(encoding="utf8", errors="replace")
    checks["log_terminal_sentinel_once"] = log_text.count("PHYSICS_GUIDED_COLLECTION_PASS 72 0") == 1
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "request_sha256": sha256(request_path),
        "lineage_sha256": sha256(folder / "lineage.json"),
        "receipt_sha256": sha256(folder / "receipt.json"),
        "output_hashes": {name: sha256(folder / name) for name in sorted(required)},
        "log_sha256": sha256(log_path),
    }


def npz_arrays(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def component_rows(arrays, metadata, thresholds):
    rows = []
    for case_index, case in enumerate(metadata["cases"]):
        valid = np.flatnonzero(arrays["valid"][:, case_index])
        ids = valid[metadata["context_steps"] :]
        body_rows = []
        total = 0
        bad = 0
        for name, geom in metadata["foot_geometry"].items():
            body = metadata["names"].index(name)
            position = arrays["actual"][ids, case_index, body]
            quaternion = arrays["quaternion"][ids, case_index, body]
            world = position[:, None] + rotate_xyzw(
                quaternion[:, None], np.asarray(geom["points"])[None]
            )
            world[:, :, 2] -= np.asarray(geom["radii"])[None]
            low = world[np.arange(len(ids)), np.argmin(world[:, :, 2], axis=1)]
            com = rotate_xyzw(
                quaternion,
                np.asarray(metadata["native_body_properties"][case_index][body]["com"]),
            )
            linear = arrays["linvel"][ids, case_index, body, :2]
            rotational = np.cross(
                arrays["angvel"][ids, case_index, body], low - position - com
            )[:, :2]
            material = linear + rotational
            support = (
                (arrays["contact_force"][ids, case_index, body, 2] > thresholds["contact_force_n"])
                & (low[:, 2] < thresholds["surface_near_ground_m"])
            )
            speed = np.linalg.norm(material, axis=1)
            count = int(support.sum())
            slip = int((support & (speed > thresholds["sliding_speed_m_s"])).sum())
            total += count
            bad += slip
            moving = support & (speed > 1e-12)
            aligned_linear = np.einsum("ij,ij->i", linear[moving], material[moving]) / speed[moving]
            aligned_rot = np.einsum("ij,ij->i", rotational[moving], material[moving]) / speed[moving]
            body_rows.append(
                {
                    "body": name,
                    "support_observations": count,
                    "slip_proxy_observations": slip,
                    "mean_COM_linear_speed_m_s": float(np.linalg.norm(linear[support], axis=1).mean()) if count else None,
                    "mean_rotational_term_speed_m_s": float(np.linalg.norm(rotational[support], axis=1).mean()) if count else None,
                    "mean_material_speed_m_s": float(speed[support].mean()) if count else None,
                    "mean_signed_linear_projection_m_s": float(aligned_linear.mean()) if moving.any() else None,
                    "mean_signed_rotational_projection_m_s": float(aligned_rot.mean()) if moving.any() else None,
                    "note": "Signed projections can cancel; term magnitudes are not additive and not standalone slip metrics",
                }
            )
        rows.append(
            {
                "case_id": case["case_id"],
                "source_id": case["source_id"],
                "support_observations": total,
                "slip_proxy_observations": bad,
                "surface_slip_proxy_fraction": bad / total if total else None,
                "bodies": body_rows,
            }
        )
    groups = [group for row in rows for group in row["bodies"] if group["support_observations"]]
    total = sum(group["support_observations"] for group in groups)

    def weighted(key):
        return (
            sum(group[key] * group["support_observations"] for group in groups) / total
            if total
            else None
        )

    return {
        "inputs": len(rows),
        "rows": rows,
        "support_observations": total,
        "pooled_slip_proxy_fraction": (
            sum(row["slip_proxy_observations"] for row in rows) / total if total else None
        ),
        "support_weighted_mean_COM_linear_speed_m_s": weighted("mean_COM_linear_speed_m_s"),
        "support_weighted_mean_rotational_term_speed_m_s": weighted("mean_rotational_term_speed_m_s"),
        "support_weighted_mean_material_speed_m_s": weighted("mean_material_speed_m_s"),
    }


def main(root: Path, output: Path):
    root = root.resolve()
    out = root / "results/support_leg_ik_20261012"
    parent = root / "results/support_reference_probe_v63_20261012"
    source_home = root / "results/paired_feedback_validation_v51_20261012"
    analysis_path = out / "analysis_1791778833277260183.json"
    velocity_paths = sorted(out.glob("velocity_components_*.json"))
    if len(velocity_paths) != 1:
        raise RuntimeError(f"Expected one component diagnosis, found {len(velocity_paths)}")
    velocity_path = velocity_paths[0]
    analysis = read_json(analysis_path)
    summary = read_json(out / "summary.json")
    thresholds = read_json(out / "frozen_thresholds.json")["thresholds"]
    versions = read_json(out / "versions.json")
    preparation = read_json(out / "preparation.json")

    current_source_hashes = {name: sha256(root / name) for name in versions["files"]}
    version_file_mismatches = {
        name: {"expected": digest, "actual": current_source_hashes[name]}
        for name, digest in versions["files"].items()
        if current_source_hashes[name] != digest
    }
    binding_checks = {
        "all_versioned_sources_match": not version_file_mismatches,
        "manifest_matches_versions": sha256(out / "source_manifest.json") == versions["source_manifest_sha256"],
        "thresholds_match_versions": sha256(out / "frozen_thresholds.json") == versions["thresholds_sha256"],
        "parent_versions_match": sha256(parent / "versions.json") == versions["parent_versions_sha256"],
        "analysis_source_lock": sha256(root / "scripts/analyze_support_leg_ik_20261012.py")
        == read_json(out / "analysis_plan_lock.json")["source_sha256"],
        "analysis_output_source_lock": analysis["source_sha256"]
        == read_json(out / "analysis_plan_lock.json")["source_sha256"],
        "state_complete": read_json(out / "state.json")["status"]
        == "COMPLETE_LIMITED_LEG_IK_DIAGNOSIS",
        "summary_complete": summary["status"] == "COMPLETE_LIMITED_LEG_IK_DIAGNOSIS",
        "analysis_complete": analysis["status"] == "COMPLETE_RAW_RECOMPUTED_LEG_IK_DIAGNOSIS",
        "pipeline_finally_zero": read_json(out / "pipeline_finally_receipt.json").get("pipeline_code") == 0,
        "external_tool_observation_exit_zero": read_json(out / "EXECUTOR_TERMINAL_OBSERVATION.json").get("observed_ssh_exit_code") == 0,
        "external_tool_observation_sentinel": read_json(out / "EXECUTOR_TERMINAL_OBSERVATION.json").get("observed_stdout_sentinel")
        == "SUPPORT_LEG_IK_ADJUDICATED_COMPLETE",
        "external_analysis_observation_exit_zero": read_json(out / "EXECUTOR_TERMINAL_OBSERVATION.json").get("observed_analysis_exit_code") == 0,
        "external_observation_honestly_labeled": read_json(out / "EXECUTOR_TERMINAL_OBSERVATION.json").get("server_automatic_exit_receipt") is False,
    }
    observation = read_json(out / "EXECUTOR_TERMINAL_OBSERVATION.json")
    observation_hash_checks = {}
    for windows_name, expected in observation["input_hashes"].items():
        normalized = windows_name.replace("\\", "/")
        observation_hash_checks[normalized] = sha256(root / normalized) == expected
    binding_checks["external_observation_inputs_current"] = all(observation_hash_checks.values())

    homes = {"baseline": parent, "cartesian": out, "two_link_ik": out}
    references = {}
    goals = {}
    for arm in homes:
        arrays = npz_arrays(out / "references" / f"{arm}.npz")
        references[arm] = arrays["reference"]
        goals[arm] = arrays["original_goal"]
    raw_arrays = {}
    metadata = {}
    stored_rows = {}
    recomputed_rows = {}
    initial_states = {}
    per_arm = {}
    row_mismatches = []
    case_orders = {}
    observation_trace_inventory = {}

    for arm, home in homes.items():
        folder = home / "raw" / f"val_{arm}_s11"
        request_path = home / "requests" / f"val_{arm}_s11.json"
        lineage = verify_batch(
            folder,
            request_path,
            out / "references" / f"{arm}.npz",
            home / "versions.json" if arm == "baseline" else out / "versions.json",
            out / "source_manifest.json",
            out / "frozen_thresholds.json",
        )
        arrays = npz_arrays(folder / "trajectories.npz")
        meta = read_json(folder / "metadata.json")
        stored = read_json(folder / "metrics.json")
        recomputed = [recompute_row(arrays, index, meta, thresholds) for index in range(72)]
        for index, (old, new) in enumerate(zip(stored, recomputed)):
            compare_value(old, new, f"{arm}.rows[{index}]", row_mismatches)
        initial = npz_arrays(folder / "initial_states.npz")
        trace = npz_arrays(folder / "observation_trace.npz")
        observation_trace_inventory[arm] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in trace.items()
        }
        initial_field_names = {
            name[len("bootstrap_") :]
            for name in initial
            if name.startswith("bootstrap_")
        }
        reset_exact = all(
            np.array_equal(initial[f"bootstrap_{field}"], initial[f"first_action_{field}"])
            for field in INITIAL_FIELDS
        )
        counts = {field: sum(int(row[field]) for row in recomputed) for field in COUNT_FIELDS}
        finite = {
            field: [
                float(row[field])
                for row in recomputed
                if row.get(field) is not None and np.isfinite(row[field])
            ]
            for field in NUMERIC_FIELDS
        }
        means = {field: float(np.mean(values)) for field, values in finite.items() if values}
        denominators = {
            field: {"finite": len(values), "missing_or_nonfinite": 72 - len(values), "total": 72}
            for field, values in finite.items()
        }
        analysis_mismatches = []
        compare_value(analysis["reports"][arm]["counts"], counts, f"analysis.{arm}.counts", analysis_mismatches)
        compare_value(analysis["reports"][arm]["means"], means, f"analysis.{arm}.means", analysis_mismatches)
        compare_value(
            analysis["reports"][arm]["metric_denominators"],
            denominators,
            f"analysis.{arm}.denominators",
            analysis_mismatches,
        )
        compare_value(summary["reports"][arm]["counts"], counts, f"summary.{arm}.counts", analysis_mismatches)
        reference_matches_raw = np.array_equal(arrays["reference"], references[arm])
        per_arm[arm] = {
            "lineage": lineage,
            "raw_row_count": len(stored),
            "recomputed_row_count": len(recomputed),
            "raw_metric_mismatch_count": sum(1 for item in row_mismatches if item["path"].startswith(arm + ".")),
            "reported_aggregate_mismatches": analysis_mismatches,
            "counts": counts,
            "means": means,
            "metric_denominators": denominators,
            "all_metric_denominators_finite_72": all(item["finite"] == 72 for item in denominators.values()),
            "metadata_seed": meta["seed"],
            "metadata_mode": meta["mode"],
            "metadata_round": meta["round"],
            "context_steps": meta["context_steps"],
            "episode_steps": meta["episode_steps"],
            "case_count": len(meta["cases"]),
            "source_count": len({row["source_id"] for row in recomputed}),
            "source_multiplicities": sorted(Counter(row["source_id"] for row in recomputed).values()),
            "reference_npz_exactly_applied": reference_matches_raw,
            "initial_state_keys": sorted(initial),
            "initial_field_names_exact": initial_field_names == INITIAL_FIELDS,
            "bootstrap_to_first_action_exact": reset_exact,
            "metadata_reset_hash_fields_exact": set(meta["manual_reset_state_sha256"]) == INITIAL_FIELDS,
            "metadata_cache_fields_exact": set(meta["initial_rigid_cache_consistency"]) == CACHE_FIELDS,
            "metadata_reset_gate_pass": bool(meta["initial_reset_gate_pass"]),
            "metadata_cache_consistency": meta["initial_rigid_cache_consistency"],
        }
        raw_arrays[arm] = arrays
        metadata[arm] = meta
        stored_rows[arm] = stored
        recomputed_rows[arm] = recomputed
        initial_states[arm] = initial
        case_orders[arm] = [row["case_id"] for row in stored]

    baseline_initial = initial_states["baseline"]
    cross_arm_initial_exact = {
        arm: set(initial) == set(baseline_initial)
        and all(np.array_equal(initial[name], baseline_initial[name]) for name in initial)
        for arm, initial in initial_states.items()
    }
    cross_arm_metadata_exact = {
        field: all(metadata[arm][field] == metadata["baseline"][field] for arm in homes)
        for field in ("original_goal", "original_heading", "cases")
    }
    cross_arm_case_order_exact = len({tuple(value) for value in case_orders.values()}) == 1

    source_folder = source_home / "raw/val_shared_s11_r0"
    source_request = source_home / "requests/val_shared_s11_r0.json"
    source_lineage = verify_batch(
        source_folder,
        source_request,
        source_home / "selected/random_s11.json",
        source_home / "versions.json",
        source_home / "source_manifest.json",
        source_home / "frozen_thresholds.json",
    )
    # The source request has no replay motion.  The generic reference check is
    # inapplicable to this construction batch and is removed from its verdict.
    source_lineage["checks"].pop("request_reference_hash")
    source_lineage["all_pass"] = all(source_lineage["checks"].values())
    source_arrays = npz_arrays(source_folder / "trajectories.npz")
    source_meta = read_json(source_folder / "metadata.json")
    source_metrics = read_json(source_folder / "metrics.json")
    constructed = np.empty_like(references["baseline"])
    source_lengths = []
    for index in range(72):
        length = int(source_arrays["valid"][:, index].sum())
        source_lengths.append(length)
        constructed[:length, index] = source_arrays["actual"][:length, index]
        constructed[length:, index] = source_arrays["actual"][length - 1, index]
    source_ids = [row["source_id"] for row in source_metrics]
    source_split_by_id = {}
    for case in source_meta["cases"]:
        source_split_by_id.setdefault(case["source_id"], case.get("official_split", "unknown"))
    official_split_clusters = Counter(source_split_by_id.values())
    official_split_inputs = Counter(case.get("official_split", "unknown") for case in source_meta["cases"])
    source_checks = {
        "execution_lineage": source_lineage["all_pass"],
        "72_inputs": len(source_metrics) == 72,
        "24_sources": len(set(source_ids)) == 24,
        "three_offsets_each": set(Counter(source_ids).values()) == {3},
        "prepared_case_order": [row["case_id"] for row in source_metrics] == preparation["source_case_order"],
        "fixed_first_construction_baseline": np.array_equal(constructed, references["baseline"]),
        "prepared_goal_hash": source_meta["original_goal_sha256"] == preparation["source_original_goal_sha256"],
        "prepared_source_binding": preparation["source_binding"] == {
            key: source_lineage[key]
            for key in ("lineage_sha256", "receipt_sha256", "request", "output_hashes", "certification")
            if key in source_lineage
        },
    }
    # Recreate the guard return schema for a direct equality test.
    source_guard_schema = {
        "lineage_sha256": source_lineage["lineage_sha256"],
        "receipt_sha256": source_lineage["receipt_sha256"],
        "request": read_json(source_request),
        "output_hashes": read_json(source_folder / "lineage.json")["outputs"],
        "certification": read_json(source_folder / "lineage.json")["certification"],
    }
    source_checks["prepared_source_binding"] = preparation["source_binding"] == source_guard_schema

    base = references["baseline"]
    cartesian = references["cartesian"]
    ik = references["two_link_ik"]
    names = metadata["baseline"]["names"]
    start = metadata["baseline"]["context_steps"]
    altered = [names.index(side + "_" + part) for side in ("L", "R") for part in ("Knee", "Ankle", "Toe")]
    unaltered = [index for index in range(len(names)) if index not in altered]
    reference_invariants = {}
    for arm, reference in references.items():
        delta = reference - base
        reference_invariants[arm] = {
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "finite": bool(np.isfinite(reference).all()),
            "prefix_0_through_context_exact": bool(np.array_equal(reference[: start + 1], base[: start + 1])),
            "last_pose_exact": bool(np.array_equal(reference[-1], base[-1])),
            "original_goals_exact": bool(np.array_equal(goals[arm], goals["baseline"])),
            "unaltered_joint_coordinates_exact": bool(np.array_equal(reference[:, :, unaltered], base[:, :, unaltered])),
            "max_any_joint_shift_m": float(np.linalg.norm(delta, axis=-1).max()),
            "incomplete_sources_unchanged": bool(
                all(
                    np.array_equal(reference[:, index], base[:, index])
                    for index, length in enumerate(source_lengths)
                    if length != 120
                )
            ),
        }
        pair_roundoff = []
        vertical_shift = []
        for side in ("L", "R"):
            ankle = names.index(side + "_Ankle")
            toe = names.index(side + "_Toe")
            pair_roundoff.append(np.max(np.abs(delta[:, :, ankle] - delta[:, :, toe])))
            vertical_shift.extend((np.max(np.abs(delta[:, :, ankle, 2])), np.max(np.abs(delta[:, :, toe, 2]))))
        reference_invariants[arm]["max_paired_ankle_toe_shift_roundoff_m"] = float(max(pair_roundoff))
        reference_invariants[arm]["max_foot_vertical_shift_m"] = float(max(vertical_shift))

    cartesian_knees_exact = all(
        np.array_equal(cartesian[:, :, names.index(side + "_Knee")], base[:, :, names.index(side + "_Knee")])
        for side in ("L", "R")
    )
    ik_foot_matches_cartesian = all(
        np.array_equal(ik[:, :, names.index(side + "_" + part)], cartesian[:, :, names.index(side + "_" + part)])
        for side in ("L", "R")
        for part in ("Ankle", "Toe")
    )
    max_bone_error = 0.0
    max_ik_knee_shift = 0.0
    max_cartesian_bone_error = 0.0
    for side in ("L", "R"):
        hip, knee, ankle = [names.index(side + "_" + part) for part in ("Hip", "Knee", "Ankle")]
        baseline_thigh = np.linalg.norm(base[:, :, knee] - base[:, :, hip], axis=-1)
        baseline_calf = np.linalg.norm(base[:, :, ankle] - base[:, :, knee], axis=-1)
        ik_thigh = np.linalg.norm(ik[:, :, knee] - ik[:, :, hip], axis=-1)
        ik_calf = np.linalg.norm(ik[:, :, ankle] - ik[:, :, knee], axis=-1)
        cart_thigh = np.linalg.norm(cartesian[:, :, knee] - cartesian[:, :, hip], axis=-1)
        cart_calf = np.linalg.norm(cartesian[:, :, ankle] - cartesian[:, :, knee], axis=-1)
        max_bone_error = max(
            max_bone_error,
            float(np.max(np.abs(ik_thigh - baseline_thigh))),
            float(np.max(np.abs(ik_calf - baseline_calf))),
        )
        max_cartesian_bone_error = max(
            max_cartesian_bone_error,
            float(np.max(np.abs(cart_thigh - baseline_thigh))),
            float(np.max(np.abs(cart_calf - baseline_calf))),
        )
        max_ik_knee_shift = max(
            max_ik_knee_shift,
            float(np.max(np.linalg.norm(ik[:, :, knee] - base[:, :, knee], axis=-1))),
        )
    reference_invariants["cartesian"]["knees_exactly_baseline"] = cartesian_knees_exact
    reference_invariants["cartesian"]["max_segment_length_error_m"] = max_cartesian_bone_error
    reference_invariants["two_link_ik"]["feet_exactly_cartesian"] = ik_foot_matches_cartesian
    reference_invariants["two_link_ik"]["max_segment_length_error_m"] = max_bone_error
    reference_invariants["two_link_ik"]["max_knee_shift_m"] = max_ik_knee_shift

    paired = {}
    for arm in ("cartesian", "two_link_ik"):
        source_delta = []
        for source_id in sorted(set(source_ids)):
            indices = [index for index, row in enumerate(recomputed_rows["baseline"]) if row["source_id"] == source_id]
            source_delta.append(
                float(
                    np.mean(
                        [
                            int(recomputed_rows[arm][index]["joint_pass"])
                            - int(recomputed_rows["baseline"][index]["joint_pass"])
                            for index in indices
                        ]
                    )
                )
            )
        source_delta_array = np.asarray(source_delta)
        bootstrap = (
            np.random.RandomState(6211)
            .choice(source_delta_array, size=(5000, 24))
            .mean(axis=1)
            * 100
        )
        recomputed_pair = {
            "joint_gain_pp": float(source_delta_array.mean() * 100),
            "source_cluster_95ci_pp": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "sources": 24,
            "resamples": 5000,
        }
        mismatch = []
        for key in recomputed_pair:
            compare_value(analysis["paired"][arm][key], recomputed_pair[key], f"paired.{arm}.{key}", mismatch)
        paired[arm] = {
            "source_level_deltas": source_delta,
            "recomputed": recomputed_pair,
            "reported_mismatches": mismatch,
        }

    component = read_json(velocity_path)
    component_mismatches = []
    recomputed_component_reports = {}
    base_ref = references["baseline"]
    reference_axis_error = {}
    for arm in homes:
        recomputed_component_reports[arm] = component_rows(raw_arrays[arm], metadata[arm], thresholds)
        compare_value(
            component["reports"][arm],
            recomputed_component_reports[arm],
            f"component.{arm}",
            component_mismatches,
        )
        errors = []
        for side in ("L", "R"):
            ankle = names.index(side + "_Ankle")
            toe = names.index(side + "_Toe")
            errors.append(
                float(
                    np.max(
                        np.abs(
                            (references[arm][:, :, toe] - references[arm][:, :, ankle])
                            - (base_ref[:, :, toe] - base_ref[:, :, ankle])
                        )
                    )
                )
            )
        reference_axis_error[arm] = max(errors)
    compare_value(
        component["reference_toe_ankle_vector_max_change_m"],
        reference_axis_error,
        "component.reference_toe_ankle_vector_max_change_m",
        component_mismatches,
    )

    current_parent_ledger = read_json(parent / "budget_ledger.json")
    child_ledger = read_json(out / "budget_ledger.json")
    historical_baseline_bytes = json.dumps([current_parent_ledger[0]], indent=2).encode("utf8")
    historical_baseline_sha = sha256_bytes(historical_baseline_bytes)
    continuation_addendum = read_json(out / "CONTINUATION_CONTRACT_ADDENDUM.json")
    ledger_checks = {
        "child_two_rows": len(child_ledger) == 2,
        "child_labels_exact": [row["label"] for row in child_ledger]
        == ["val_cartesian_s11", "val_two_link_ik_s11"],
        "child_all_exit_zero": all(row["exit_code"] == 0 for row in child_ledger),
        "child_request_hashes": all(
            row["request_sha256"] == sha256(out / "requests" / f"{row['label']}.json")
            for row in child_ledger
        ),
        "current_parent_three_rows": len(current_parent_ledger) == 3,
        "current_parent_hash": sha256(parent / "budget_ledger.json")
        == continuation_addendum["ledger_observation"]["current_whole_ledger_sha256"],
        "historical_baseline_hash_reconstructed": historical_baseline_sha
        == continuation_addendum["ledger_observation"]["historical_whole_ledger_sha256"],
        "baseline_row_unchanged": current_parent_ledger[0]
        == continuation_addendum["ledger_observation"]["rows"][0],
    }
    costs = {
        "new_arm_seconds": {row["label"]: row["seconds"] for row in child_ledger},
        "new_arm_total_seconds": sum(row["seconds"] for row in child_ledger),
        "cached_baseline_original_seconds": current_parent_ledger[0]["seconds"],
        "current_parent_all_three_arm_seconds": sum(row["seconds"] for row in current_parent_ledger),
        "separate_saved_array_adjudication_seconds": sum(
            row["seconds"] for row in read_json(parent / "numerical_adjudication_budget_ledger.json")
        ),
    }

    witness = read_json(out / "DOCUMENT_WITNESS.json")
    continuation_witness = read_json(out / "CONTINUATION_DOCUMENT_WITNESS.json")
    adjudicated_witness = read_json(out / "ADJUDICATED_DOCUMENT_WITNESS.json")
    scalar_diagnosis = read_json(out / "SCALAR_CONTRACT_DIAGNOSIS.json")
    adjudication = continuation_addendum
    witness_checks = {
        "original_fail_retained": witness["status"] == "FAIL",
        "original_fail_only_logged_scalar_checks": all(
            item["check"].startswith("frozen_proxy_objective_records_") for item in witness["failures"]
        ),
        "scalar_contract_independent_pass": scalar_diagnosis["status"]
        == "PASS_DOCUMENTED_PREPARATION_INVARIANTS"
        and all(scalar_diagnosis["checks"].values()),
        "continuation_fail_retained": continuation_witness["status"] == "FAIL",
        "continuation_fail_only_historical_ledger": [item["check"] for item in continuation_witness["failures"]]
        == ["cached_parent_budget_ledger.json"],
        "continuation_contract_independent_pass": adjudication["status"]
        == "PASS_DOCUMENTED_CONTINUATION_INVARIANTS"
        and all(adjudication["checks"].values()),
        "adjudicated_witness_pass": adjudicated_witness["status"] == "PASS"
        and all(adjudicated_witness["checks"].values()),
    }

    analysis_bindings_current = {
        name: sha256(root / name) == digest for name, digest in analysis["input_hashes"].items()
    }
    component_bindings_current = {
        name: sha256(root / name) == digest for name, digest in component["input_hashes"].items()
    }

    analysis_decision_expected = (
        "EXPLORATORY_JOINT_GAIN_NOT_INNOVATION_ACCEPTANCE"
        if any(paired[arm]["recomputed"]["joint_gain_pp"] > 0 for arm in paired)
        else "PHYSICAL_COMPONENT_GAIN_NO_JOINT_EFFICACY"
        if any(
            per_arm[arm]["counts"]["physical_pass"] > per_arm["baseline"]["counts"]["physical_pass"]
            for arm in ("cartesian", "two_link_ik")
        )
        else "STOP_THIS_LIMITED_LEG_TRANSLATION_AND_POSITION_IK_PARAMETERIZATION"
    )

    audit_hash_paths = [
        out / "summary.json",
        out / "state.json",
        analysis_path,
        out / "analysis_plan_lock.json",
        out / "preparation.json",
        out / "versions.json",
        out / "source_manifest.json",
        out / "frozen_thresholds.json",
        out / "budget_ledger.json",
        out / "launch_receipt.json",
        out / "pipeline_finally_receipt.json",
        out / "EXECUTOR_TERMINAL_OBSERVATION.json",
        out / "preflight.json",
        out / "continuation_preflight.json",
        out / "adjudicated_preflight.json",
        out / "DOCUMENT_WITNESS.json",
        out / "SCALAR_CONTRACT_DIAGNOSIS.json",
        out / "CONTINUATION_DOCUMENT_WITNESS.json",
        out / "CONTINUATION_CONTRACT_ADDENDUM.json",
        out / "ADJUDICATED_DOCUMENT_WITNESS.json",
        velocity_path,
    ]
    for arm, home in homes.items():
        folder = home / "raw" / f"val_{arm}_s11"
        audit_hash_paths.extend(
            [
                folder / name
                for name in (
                    "metadata.json",
                    "metrics.json",
                    "receipt.json",
                    "lineage.json",
                    "trajectories.npz",
                    "initial_states.npz",
                    "observation_trace.npz",
                )
            ]
        )
        audit_hash_paths.append(home / "requests" / f"val_{arm}_s11.json")
        audit_hash_paths.append(out / "references" / f"{arm}.npz")

    all_checks = {
        "binding_checks": all(binding_checks.values()),
        "version_sources": not version_file_mismatches,
        "lineage_all_arms": all(arm["lineage"]["all_pass"] for arm in per_arm.values()),
        "raw_rows_match": not row_mismatches,
        "aggregates_match": all(not arm["reported_aggregate_mismatches"] for arm in per_arm.values()),
        "all_denominators_finite": all(arm["all_metric_denominators_finite_72"] for arm in per_arm.values()),
        "all_sources_three_offsets": all(
            arm["source_count"] == 24 and set(arm["source_multiplicities"]) == {3}
            for arm in per_arm.values()
        ),
        "cross_arm_metadata": all(cross_arm_metadata_exact.values()) and cross_arm_case_order_exact,
        "initial_state_exports": all(cross_arm_initial_exact.values())
        and all(arm["bootstrap_to_first_action_exact"] for arm in per_arm.values())
        and all(all(value == 0 for value in arm["metadata_cache_consistency"].values()) for arm in per_arm.values()),
        "source_construction": all(source_checks.values()),
        "reference_prefix_endpoint_targets": all(
            entry["prefix_0_through_context_exact"]
            and entry["last_pose_exact"]
            and entry["original_goals_exact"]
            and entry["finite"]
            for entry in reference_invariants.values()
        ),
        "reference_shift_bounds": all(
            entry["max_any_joint_shift_m"] <= 0.10001
            and entry["max_foot_vertical_shift_m"] == 0.0
            and entry["max_paired_ankle_toe_shift_roundoff_m"] <= 1e-6
            for entry in reference_invariants.values()
        ),
        "two_link_bones": max_bone_error <= 1e-5 and max_ik_knee_shift <= 0.10001,
        "cartesian_control": cartesian_knees_exact and ik_foot_matches_cartesian,
        "bootstrap_match": all(not item["reported_mismatches"] for item in paired.values()),
        "component_reproduction": not component_mismatches,
        "ledger_semantics": all(ledger_checks.values()),
        "witness_history": all(witness_checks.values()),
        "analysis_bindings_current": all(analysis_bindings_current.values()),
        "component_bindings_current": all(component_bindings_current.values()),
        "decision_reproduced": analysis["decision"] == analysis_decision_expected,
    }

    result = {
        "status": "PASS_INDEPENDENT_RAW_RECOMPUTATION" if all(all_checks.values()) else "FAIL_INDEPENDENT_RAW_RECOMPUTATION",
        "audit_script": "independent_recompute.py",
        "python": {"numpy": np.__version__},
        "root": str(root),
        "checks": all_checks,
        "binding_checks": binding_checks,
        "external_observation_hash_checks": observation_hash_checks,
        "version_file_mismatches": version_file_mismatches,
        "per_arm": per_arm,
        "raw_metric_mismatches": row_mismatches,
        "cross_arm_initial_state_exact": cross_arm_initial_exact,
        "cross_arm_metadata_exact": cross_arm_metadata_exact,
        "cross_arm_case_order_exact": cross_arm_case_order_exact,
        "source_construction": {
            "checks": source_checks,
            "lineage": source_lineage,
            "valid_lengths": dict(Counter(source_lengths)),
            "official_split_clusters": dict(official_split_clusters),
            "official_split_inputs": dict(official_split_inputs),
        },
        "reference_invariants": reference_invariants,
        "paired_source_cluster_bootstrap": paired,
        "component_diagnosis": {
            "path": str(velocity_path.relative_to(root)),
            "mismatches": component_mismatches,
            "pooled": {
                arm: {key: value for key, value in report.items() if key != "rows"}
                for arm, report in recomputed_component_reports.items()
            },
            "reference_toe_ankle_vector_max_change_m": reference_axis_error,
        },
        "ledger": {
            "checks": ledger_checks,
            "historical_baseline_only_sha256": historical_baseline_sha,
            "current_parent_ledger_sha256": sha256(parent / "budget_ledger.json"),
            "costs": costs,
        },
        "witness_contracts": {
            "checks": witness_checks,
            "original_failure_count": len(witness["failures"]),
            "continuation_failure_count": len(continuation_witness["failures"]),
            "original_logged_norm_max_abs_difference_m": 6.938893903907228e-17,
            "note": "The float64 logged-norm differences are retained historical evidence; the saved float32 references and documented geometric inequalities are the operative invariants.",
        },
        "analysis_bindings_current": analysis_bindings_current,
        "component_bindings_current": component_bindings_current,
        "reported_decision": analysis["decision"],
        "recomputed_decision": analysis_decision_expected,
        "observation_trace_inventory": observation_trace_inventory,
        "limitations": [
            "Exported reset arrays and four recorded rigid-cache checks match; hidden PhysX engine state is not established.",
            "The 24 source clusters mix upstream official train and test provenance inside one internal validation slice; pretrained exposure was not independently established.",
            "The evaluation is one-seed simulation-only and uses collision-geometry/body-force and model-derived tracking proxies.",
            "The external tool observation records terminal exit/sentinels but is explicitly not an autonomous server parent-exit receipt.",
        ],
        "audited_input_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in audit_hash_paths
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(result["status"])
    print(json.dumps({"checks": all_checks, "counts": {arm: item["counts"] for arm, item in per_arm.items()}, "costs": costs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.output)
