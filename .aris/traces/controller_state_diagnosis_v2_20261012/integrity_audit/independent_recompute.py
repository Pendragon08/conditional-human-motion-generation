"""Read-only independent recomputation for the controller-state integrity audit.

The script does not import the experiment metric implementation. It reads frozen
artifacts, implements the documented equations independently, and writes only an
optional audit-trace JSON outside the experiment result payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/code/ouyanghao/oyh/conditional-human-motion-generation")
OUT = ROOT / "results/controller_state_diagnosis_v2_20261012"
STATES = ("multitask", "t2m", "unfinetuned")
NUMERIC_MEANS = (
    "endpoint_error_m",
    "heading_error_rad",
    "surface_slip_proxy_fraction",
    "penetration_fraction",
    "jerk_p95_m_s3",
    "tracking_mean_m",
)
CLASSIFICATIONS = (
    "completed",
    "condition_pass",
    "physical_pass",
    "tracking_pass",
    "joint_pass",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(archive: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(archive.files):
        value = np.ascontiguousarray(archive[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def wrap_angle(value: Any) -> Any:
    return (value + np.pi) % (2 * np.pi) - np.pi


def body_heading(positions: np.ndarray, names: list[str]) -> np.ndarray:
    across = (
        positions[..., names.index("R_Hip"), :]
        - positions[..., names.index("L_Hip"), :]
        + positions[..., names.index("R_Shoulder"), :]
        - positions[..., names.index("L_Shoulder"), :]
    )
    forward = np.cross(np.array([0.0, 0.0, 1.0]), across)
    return np.arctan2(forward[..., 1], forward[..., 0])


def rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    imaginary, real = quaternion[..., :3], quaternion[..., 3:]
    return vector + 2 * np.cross(
        imaginary, np.cross(imaginary, vector) + real * vector
    )


def independent_surface_metrics(
    position: np.ndarray,
    quaternion: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    force: np.ndarray,
    names: list[str],
    geometry: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    speeds, support, heights = [], [], []
    for name, specification in geometry.items():
        body_index = names.index(name)
        points = np.array(specification["points"], dtype=float)
        radii = np.array(specification["radii"], dtype=float)
        rotated = rotate_xyzw(quaternion[:, body_index, None, :], points[None])
        world = position[:, body_index, None, :] + rotated
        world[:, :, 2] -= radii[None]
        closest = np.argmin(world[:, :, 2], axis=1)
        lowest = world[np.arange(len(position)), closest]
        lever = lowest - position[:, body_index]
        center_of_mass = rotate_xyzw(
            quaternion[:, body_index],
            np.asarray(specification["com_local"], dtype=float),
        )
        material_velocity = linear_velocity[:, body_index] + np.cross(
            angular_velocity[:, body_index], lever - center_of_mass
        )
        supported = (
            force[:, body_index, 2] > thresholds["contact_force_n"]
        ) & (lowest[:, 2] < thresholds["surface_near_ground_m"])
        speeds.append(np.linalg.norm(material_velocity[:, :2], axis=1))
        support.append(supported)
        heights.append(lowest[:, 2])
    speed = np.stack(speeds, axis=1)
    is_supported = np.stack(support, axis=1)
    height = np.stack(heights, axis=1)
    count = int(is_supported.sum())
    return {
        "surface_slip_proxy_mean_m_s": (
            float(speed[is_supported].mean()) if count else None
        ),
        "surface_slip_proxy_fraction": (
            float(
                (
                    speed[is_supported] > thresholds["sliding_speed_m_s"]
                ).mean()
            )
            if count
            else None
        ),
        "support_observations": count,
        "penetration_fraction": float(
            (height < -thresholds["penetration_depth_m"]).any(axis=1).mean()
        ),
        "penetration_depth_p95_m": float(
            np.quantile(np.maximum(-height, 0), 0.95)
        ),
        "airborne_fraction": float((~is_supported.any(axis=1)).mean()),
        "contact_metric_type": (
            "COM velocity corrected body-force and collision-primitive surface proxy"
        ),
    }


def independent_metrics(
    arrays: dict[str, np.ndarray],
    index: int,
    metadata: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = arrays["valid"][:, index]
    length = int(valid.sum())
    expected = int(metadata["episode_steps"])
    names = metadata["names"]
    dt = float(metadata["dt"])
    target = np.asarray(metadata["original_goal"][index])
    joint = names.index(metadata["cases"][index]["goal_joint_body"])
    goal_heading = float(metadata["original_heading"][index])
    result = {
        key: metadata["cases"][index][key]
        for key in ("case_id", "source_id", "clip_id", "family", "offset_m")
    }
    result.update(
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
        result.update(joint_pass=False, status="no_valid_steps")
        return result, {}
    position = arrays["actual"][valid, index]
    reference = arrays["reference"][valid, index]
    distances = np.linalg.norm(position[:, joint] - target, axis=-1)
    result["endpoint_error_m"] = float(distances[-1])
    result["endpoint_error_vector"] = (target - position[-1, joint]).tolist()
    result["heading_error_rad"] = float(
        abs(wrap_angle(body_heading(position[-1], names) - goal_heading))
    )
    result["heading_error_signed_rad"] = float(
        wrap_angle(goal_heading - body_heading(position[-1], names))
    )
    prediction_start = int(metadata["context_steps"])
    near_indices: np.ndarray
    if length > prediction_start:
        generated_distances = distances[prediction_start:]
        near_indices = (
            np.flatnonzero(
                generated_distances <= generated_distances.min() + 0.001
            )
            + prediction_start
        )
        best = near_indices[
            np.argmin(np.abs(near_indices - (expected - 1)))
        ]
        result["endpoint_time_error_s"] = float(
            abs(best - (expected - 1)) * dt
        )
    else:
        near_indices = np.array([], dtype=int)
        best = None
        result["endpoint_time_error_s"] = expected * dt
    start = min(prediction_start, length - 1)
    tracking_errors = np.linalg.norm(
        position[start:] - reference[start:], axis=-1
    )
    result["tracking_mean_m"] = float(tracking_errors.mean())
    result["tracking_p95_m"] = float(
        np.quantile(tracking_errors.mean(axis=1), 0.95)
    )
    velocity = np.diff(position[start:], axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    result["jerk_p95_m_s3"] = (
        float(np.quantile(np.linalg.norm(jerk, axis=-1), 0.95))
        if len(jerk)
        else None
    )
    geometry = {
        name: dict(
            specification,
            com_local=metadata["native_body_properties"][index][
                names.index(name)
            ]["com"],
        )
        for name, specification in metadata["foot_geometry"].items()
    }
    result.update(
        independent_surface_metrics(
            position[start:],
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
        result["endpoint_error_m"] <= thresholds["goal_position_m"]
        and result["heading_error_rad"] <= thresholds["heading_rad"]
        and result["endpoint_time_error_s"]
        <= thresholds["goal_time_s"] + 1e-6
    )
    physical = (
        result["penetration_fraction"] <= thresholds["penetration_fraction"]
        and result["surface_slip_proxy_fraction"] is not None
        and result["surface_slip_proxy_fraction"]
        <= thresholds["sliding_fraction"]
        and result["jerk_p95_m_s3"] is not None
        and result["jerk_p95_m_s3"]
        <= thresholds["continuity_jerk_p95_m_s3"]
    )
    tracking = (
        result["tracking_mean_m"] <= thresholds["tracking_mean_m"]
        and result["tracking_p95_m"] <= thresholds["tracking_p95_m"]
    )
    result.update(
        condition_pass=bool(condition),
        physical_pass=bool(physical),
        tracking_pass=bool(tracking),
        joint_pass=bool(result["completed"] and condition and physical and tracking),
        status="measured",
        semantic_status="not_verified_by_independent_evaluator",
    )
    entered = np.flatnonzero(
        distances[prediction_start:] <= thresholds["goal_position_m"]
    ) + prediction_start
    semantics = {
        "nearest_distance_m": float(distances[prediction_start:].min())
        if length > prediction_start
        else None,
        "near_minimum_latest_index": int(best) if best is not None else None,
        "near_minimum_candidate_count": int(len(near_indices)),
        "entered_position_threshold": bool(len(entered)),
        "first_position_threshold_index": int(entered[0]) if len(entered) else None,
        "final_position_threshold_pass": bool(
            distances[-1] <= thresholds["goal_position_m"]
        ),
        "heading_threshold_pass": bool(
            result["heading_error_rad"] <= thresholds["heading_rad"]
        ),
        "near_minimum_time_proxy_pass": bool(
            result["endpoint_time_error_s"]
            <= thresholds["goal_time_s"] + 1e-6
        ),
    }
    return result, semantics


def compare_values(
    actual: Any,
    expected: Any,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            differences.append(
                {
                    "path": path,
                    "kind": "keys",
                    "actual": sorted(actual),
                    "expected": sorted(expected),
                }
            )
            return
        for key in actual:
            compare_values(actual[key], expected[key], f"{path}.{key}", differences)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            differences.append(
                {
                    "path": path,
                    "kind": "length",
                    "actual": len(actual),
                    "expected": len(expected),
                }
            )
            return
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_values(left, right, f"{path}[{index}]", differences)
        return
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None:
            if actual != expected:
                differences.append(
                    {"path": path, "kind": "none", "actual": actual, "expected": expected}
                )
            return
        left, right = float(actual), float(expected)
        tolerance = 64 * np.finfo(np.float64).eps * max(1.0, abs(left), abs(right))
        if not (math.isfinite(left) and math.isfinite(right)) or abs(left - right) > tolerance:
            differences.append(
                {
                    "path": path,
                    "kind": "float",
                    "actual": left,
                    "expected": right,
                    "absolute_difference": abs(left - right),
                    "tolerance": tolerance,
                }
            )
        return
    if actual != expected:
        differences.append(
            {"path": path, "kind": "discrete", "actual": actual, "expected": expected}
        )


def independent_failure_components(
    rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    pairs = {
        "position": ("endpoint_error_m", "goal_position_m", 0.0),
        "heading": ("heading_error_rad", "heading_rad", 0.0),
        "time_proxy": ("endpoint_time_error_s", "goal_time_s", 1e-6),
        "penetration_proxy": (
            "penetration_fraction",
            "penetration_fraction",
            0.0,
        ),
        "surface_slip_proxy": (
            "surface_slip_proxy_fraction",
            "sliding_fraction",
            0.0,
        ),
        "continuity_proxy": (
            "jerk_p95_m_s3",
            "continuity_jerk_p95_m_s3",
            0.0,
        ),
        "tracking_mean_proxy": (
            "tracking_mean_m",
            "tracking_mean_m",
            0.0,
        ),
        "tracking_p95_proxy": (
            "tracking_p95_m",
            "tracking_p95_m",
            0.0,
        ),
    }
    failures = {key: 0 for key in pairs}
    physical_rows = []
    for row in rows:
        failed = []
        for name, (metric, threshold, tolerance) in pairs.items():
            value = row.get(metric)
            if (
                value is None
                or not math.isfinite(value)
                or value > thresholds[threshold] + tolerance
            ):
                failures[name] += 1
                failed.append(name)
        if not row["completed"]:
            failed.append("completion")
        if row["physical_pass"]:
            physical_rows.append(
                {
                    "case_id": row["case_id"],
                    "source_id": row["source_id"],
                    "failed_frozen_criteria": failed,
                    "endpoint_error_m": row["endpoint_error_m"],
                    "heading_error_rad": row["heading_error_rad"],
                    "endpoint_time_error_s": row["endpoint_time_error_s"],
                    "joint_pass": row["joint_pass"],
                }
            )
    return {
        "inputs": len(rows),
        "criterion_failure_counts": failures,
        "completion_failures": sum(not row["completed"] for row in rows),
        "physically_passing_rows": physical_rows,
    }


def main() -> dict[str, Any]:
    versions = read_json(OUT / "versions.json")
    analysis = read_json(OUT / "analysis.json")
    thresholds = read_json(OUT / "frozen_thresholds.json")["thresholds"]
    preparation = read_json(OUT / "preparation.json")
    ledger = read_json(OUT / "budget_ledger.json")
    failure_components = read_json(OUT / "failure_components.json")
    source_hash_mismatches = {
        name: {"expected": expected, "actual": sha256(ROOT / name)}
        for name, expected in versions["files"].items()
        if sha256(ROOT / name) != expected
    }
    audited_hashes: dict[str, str] = {}
    for path in (
        OUT / "analysis.json",
        OUT / "versions.json",
        OUT / "preparation.json",
        OUT / "source_manifest.json",
        OUT / "frozen_thresholds.json",
        OUT / "runtime_witness.json",
        OUT / "DOCUMENT_WITNESS.json",
        OUT / "EXECUTOR_TERMINAL_OBSERVATION.json",
        OUT / "budget_ledger.json",
        OUT / "failure_components.json",
        ROOT / "results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json",
        ROOT
        / ".aris/traces/controller_state_diagnosis_v2_20261012/document_witness/artifact_verifier_attempt1_DOCUMENT_WITNESS.json",
    ):
        audited_hashes[str(path.relative_to(ROOT))] = sha256(path)
    manifest = read_json(OUT / "source_manifest.json")
    val_rows = [row for row in manifest["rows"] if row["split"] == "val"]
    manifest_summary = {
        "rows": len(manifest["rows"]),
        "validation_sources": len(val_rows),
        "validation_families": dict(Counter(row["family"] for row in val_rows)),
        "validation_official_splits": dict(
            Counter(row["official_split"] for row in val_rows)
        ),
        "source_ids_unique": len({row["source_id"] for row in val_rows}),
    }
    with np.load(OUT / "references/fixed.npz", allow_pickle=False) as reference_file:
        fixed_reference = {key: reference_file[key] for key in reference_file.files}
        reference_content_hash = content_hash(reference_file)
        reference_keys = list(reference_file.files)
    support_root = ROOT / "results/support_reference_probe_v63_20261012"
    support_preparation = read_json(support_root / "preparation.json")
    support_reference_path = support_root / "references/baseline.npz"
    support_source_root = (
        ROOT / "results/paired_feedback_validation_v51_20261012/raw/val_shared_s11_r0"
    )
    support_source_path = support_source_root / "trajectories.npz"
    support_source_metadata = read_json(support_source_root / "metadata.json")
    with np.load(support_reference_path, allow_pickle=False) as archive:
        support_reference = {key: archive[key] for key in archive.files}
    with np.load(support_source_path, allow_pickle=False) as archive:
        support_source_actual = archive["actual"]
        support_source_valid = archive["valid"]
    reconstructed_support_reference = np.empty_like(support_source_actual)
    for case_index in range(support_source_actual.shape[1]):
        valid_length = int(support_source_valid[:, case_index].sum())
        reconstructed_support_reference[:valid_length, case_index] = (
            support_source_actual[:valid_length, case_index]
        )
        reconstructed_support_reference[valid_length:, case_index] = (
            support_source_actual[valid_length - 1, case_index]
        )
    reference_provenance_checks = {
        "fixed_file_exactly_matches_support_baseline": sha256(
            OUT / "references/fixed.npz"
        )
        == sha256(support_reference_path),
        "support_baseline_hash_matches_parent_preparation": sha256(
            support_reference_path
        )
        == support_preparation["baseline_reference_sha256"],
        "support_baseline_exactly_reconstructs_prior_simulated_actual_with_terminal_padding": np.array_equal(
            support_reference["reference"], reconstructed_support_reference
        ),
        "support_baseline_goal_matches_prior_simulation_metadata": np.array_equal(
            support_reference["original_goal"],
            np.asarray(support_source_metadata["original_goal"], dtype=np.float32),
        ),
        "parent_declares_no_new_simulation_or_generation_during_reference_preparation": support_preparation[
            "simulation_calls"
        ]
        == 0
        and support_preparation["generation_calls"] == 0,
    }
    audited_hashes[
        "results/controller_state_diagnosis_v2_20261012/references/fixed.npz"
    ] = sha256(OUT / "references/fixed.npz")
    for path in (
        support_root / "preparation.json",
        support_reference_path,
        support_source_root / "metadata.json",
        support_source_path,
    ):
        audited_hashes[str(path.relative_to(ROOT))] = sha256(path)
    original_failure = ROOT / "results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json"
    wrong_axis_failure = (
        ROOT
        / ".aris/traces/controller_state_diagnosis_v2_20261012/document_witness/artifact_verifier_attempt1_DOCUMENT_WITNESS.json"
    )
    preservation = {
        "original_preflight_failure_status": read_json(original_failure)["status"],
        "original_preflight_failure_hash_matches_versions": sha256(original_failure)
        == versions["original_failed_witness_sha256"],
        "wrong_axis_failure_status": read_json(wrong_axis_failure)["status"],
        "wrong_axis_failure_preserved": wrong_axis_failure.exists(),
    }
    document_witness = read_json(OUT / "DOCUMENT_WITNESS.json")
    terminal_transcript = (
        ROOT
        / ".aris/traces/controller_state_diagnosis_v2_20261012/document_witness/terminal_full.txt"
    )
    terminal_observation = read_json(OUT / "EXECUTOR_TERMINAL_OBSERVATION.json")
    terminal_checks = {
        "document_witness_status": document_witness["status"],
        "document_check_exit_zero": document_witness["terminal_exit_code"] == 0,
        "document_check_invocations": document_witness["check_invocations"],
        "document_check_transcript_hash_matches": sha256(terminal_transcript)
        == document_witness["tool_evidence"]["terminal_transcript_sha256"],
        "document_check_sentinel_present": "CONTROLLER_STATE_DOCUMENT_CHECK_PASS"
        in terminal_transcript.read_text(encoding="utf8"),
        "run_external_observation_status": terminal_observation["status"],
        "run_external_observation_exit_zero": terminal_observation[
            "actual_parent_exit_code"
        ]
        == 0,
        "run_external_observation_server_generated": terminal_observation[
            "automatic_server_parent_receipt"
        ],
        "run_external_observation_sentinel": terminal_observation["sentinel"],
    }
    state_reports: dict[str, Any] = {}
    initial_archives: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_state: dict[str, dict[str, Any]] = {}
    trajectory_references: dict[str, np.ndarray] = {}
    metric_rows_by_state: dict[str, list[dict[str, Any]]] = {}
    all_metric_differences: list[dict[str, Any]] = []
    binding_failures: list[str] = []
    binding_failures.extend(
        f"reference_provenance:{name}"
        for name, passed in reference_provenance_checks.items()
        if not passed
    )
    for state in STATES:
        raw = OUT / "raw" / state
        metadata = read_json(raw / "metadata.json")
        saved_rows = read_json(raw / "metrics.json")
        request = read_json(OUT / "requests" / f"{state}.json")
        lineage = read_json(raw / "lineage.json")
        receipt = read_json(raw / "receipt.json")
        loaded_input = read_json(raw / "loaded_input_binding.json")
        loaded_checkpoint = read_json(raw / "loaded_checkpoint_binding.json")
        parent_return = read_json(OUT / f"{state}_parent_return.json")
        with np.load(raw / "trajectories.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with np.load(raw / "initial_states.npz", allow_pickle=False) as archive:
            initial = {key: archive[key] for key in archive.files}
        initial_archives[state] = initial
        metadata_by_state[state] = metadata
        trajectory_references[state] = arrays["reference"]
        metric_rows_by_state[state] = saved_rows
        computed_rows, endpoint_rows = [], []
        for index in range(72):
            computed, endpoint_semantics = independent_metrics(
                arrays, index, metadata, thresholds
            )
            computed_rows.append(computed)
            endpoint_rows.append(endpoint_semantics)
            differences: list[dict[str, Any]] = []
            compare_values(computed, saved_rows[index], f"{state}[{index}]", differences)
            all_metric_differences.extend(differences)
        counts = {
            key: sum(int(row[key]) for row in computed_rows)
            for key in CLASSIFICATIONS
        }
        means, denominators = {}, {}
        for key in NUMERIC_MEANS:
            values = [
                row[key]
                for row in computed_rows
                if row.get(key) is not None and math.isfinite(row[key])
            ]
            means[key] = float(np.mean(values)) if values else None
            denominators[key] = {"finite": len(values), "total": 72}
        aggregate_differences: list[dict[str, Any]] = []
        compare_values(
            counts,
            analysis["reports"][state]["counts"],
            f"analysis.{state}.counts",
            aggregate_differences,
        )
        compare_values(
            means,
            analysis["reports"][state]["means"],
            f"analysis.{state}.means",
            aggregate_differences,
        )
        compare_values(
            denominators,
            analysis["reports"][state]["denominators"],
            f"analysis.{state}.denominators",
            aggregate_differences,
        )
        all_metric_differences.extend(aggregate_differences)
        expected_reference = fixed_reference["reference"]
        reference_delta = float(
            np.max(np.abs(arrays["reference"][arrays["valid"]] - expected_reference[arrays["valid"]]))
        )
        endpoint_summary = {
            "final_position_pass": sum(
                row["final_position_threshold_pass"] for row in endpoint_rows
            ),
            "heading_pass": sum(row["heading_threshold_pass"] for row in endpoint_rows),
            "near_minimum_time_proxy_pass": sum(
                row["near_minimum_time_proxy_pass"] for row in endpoint_rows
            ),
            "entered_position_threshold": sum(
                row["entered_position_threshold"] for row in endpoint_rows
            ),
            "final_position_pass_but_time_proxy_fail": sum(
                row["final_position_threshold_pass"]
                and not row["near_minimum_time_proxy_pass"]
                for row in endpoint_rows
            ),
            "time_proxy_pass_but_final_position_fail": sum(
                row["near_minimum_time_proxy_pass"]
                and not row["final_position_threshold_pass"]
                for row in endpoint_rows
            ),
            "position_and_heading_pass_but_time_proxy_fail": sum(
                row["final_position_threshold_pass"]
                and row["heading_threshold_pass"]
                and not row["near_minimum_time_proxy_pass"]
                for row in endpoint_rows
            ),
        }
        request_hash = sha256(OUT / "requests" / f"{state}.json")
        expected_weight_hashes = {
            versions["generator_weights_sha256"],
            versions["controller_states"][state]["sha256"],
        }
        loaded_weight_hashes = set(
            loaded_checkpoint["actual_loaded_weight_bytes"].values()
        )
        checks = {
            "request_equals_lineage": request == lineage["request"],
            "request_hash_matches_ledger": any(
                row["label"] == state and row["request_sha256"] == request_hash
                for row in ledger
            ),
            "request_hash_matches_loaded_checkpoint": request_hash
            == loaded_checkpoint["request_sha256"],
            "versions_hash_matches_request_and_bindings": request["versions"]
            == loaded_input["versions_sha256"]
            == loaded_checkpoint["versions_sha256"]
            == sha256(OUT / "versions.json"),
            "loaded_replay_content_hash_matches": loaded_input[
                "loaded_replay_content_sha256"
            ]
            == request["replay_motion"]
            == reference_content_hash,
            "loaded_weight_hash_set_matches": loaded_weight_hashes
            == expected_weight_hashes,
            "lineage_output_hashes_match": all(
                sha256(raw / name) == digest
                for name, digest in lineage["outputs"].items()
            ),
            "receipt_matches": receipt["cases"] == 72
            and receipt["completed"] == counts["completed"]
            and receipt["joint_pass"] == counts["joint_pass"]
            and receipt["trajectories_sha256"]
            == sha256(raw / "trajectories.npz"),
            "parent_child_return_zero": parent_return["child_returncode"] == 0,
            "metadata_generation_calls_zero": metadata["generation_batch_calls"] == 0,
            "reset_gate": metadata["initial_reset_gate_pass"] is True,
            "rigid_cache_exact": all(
                value == 0 for value in metadata["initial_rigid_cache_consistency"].values()
            ),
            "bootstrap_first_action_exact": all(
                np.array_equal(
                    initial[f"bootstrap_{key}"], initial[f"first_action_{key}"]
                )
                for key in (
                    "_humanoid_root_states",
                    "_dof_pos",
                    "_dof_vel",
                    "_rigid_body_pos",
                    "_rigid_body_rot",
                    "_rigid_body_vel",
                    "_rigid_body_ang_vel",
                )
            ),
        }
        for name, passed in checks.items():
            if not passed:
                binding_failures.append(f"{state}:{name}")
        for name in (
            "trajectories.npz",
            "initial_states.npz",
            "metadata.json",
            "metrics.json",
            "lineage.json",
            "receipt.json",
            "loaded_checkpoint_binding.json",
            "loaded_input_binding.json",
        ):
            relative = str((raw / name).relative_to(ROOT))
            audited_hashes[relative] = sha256(raw / name)
            if analysis["input_hashes"].get(relative) != audited_hashes[relative]:
                binding_failures.append(f"{state}:analysis_input_hash:{name}")
        state_reports[state] = {
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "initial_shapes": {key: list(value.shape) for key, value in initial.items()},
            "valid_lengths": [int(value) for value in arrays["valid"].sum(axis=0)],
            "terminated_count": int(arrays["terminated"].sum()),
            "counts": counts,
            "means": means,
            "denominators": denominators,
            "reference_max_abs_error_to_fixed": reference_delta,
            "endpoint_semantics": endpoint_summary,
            "bindings": checks,
            "metric_rows_exact_within_float64_roundoff": not any(
                item["path"].startswith(f"{state}[")
                for item in all_metric_differences
            ),
        }
    base_initial = initial_archives[STATES[0]]
    initial_cross_state_exact = all(
        set(initial_archives[state]) == set(base_initial)
        and all(
            np.array_equal(initial_archives[state][key], base_initial[key])
            for key in base_initial
        )
        for state in STATES[1:]
    )
    metadata_cross_state = {
        key: all(metadata_by_state[state][key] == metadata_by_state[STATES[0]][key] for state in STATES[1:])
        for key in ("cases", "original_goal", "original_heading")
    }
    common_valid = np.logical_and.reduce(
        [
            np.load(OUT / "raw" / state / "trajectories.npz", allow_pickle=False)[
                "valid"
            ]
            for state in STATES
        ]
    )
    common_reference_max = 0.0
    for state in STATES[1:]:
        if common_valid.any():
            common_reference_max = max(
                common_reference_max,
                float(
                    np.max(
                        np.abs(
                            trajectory_references[state][common_valid]
                            - trajectory_references[STATES[0]][common_valid]
                        )
                    )
                ),
            )
    expected_failure_reports = {
        state: independent_failure_components(metric_rows_by_state[state], thresholds)
        for state in STATES
    }
    failure_differences: list[dict[str, Any]] = []
    compare_values(
        expected_failure_reports,
        failure_components["reports"],
        "failure_components.reports",
        failure_differences,
    )
    expected_failure_hashes = {
        "thresholds": sha256(OUT / "frozen_thresholds.json"),
        **{
            state: sha256(OUT / "raw" / state / "metrics.json")
            for state in STATES
        },
    }
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/CLoSD"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    upstream_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT / "third_party/CLoSD"),
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "closd",
        ],
        text=True,
    )
    weight_checks = {}
    for state, specification in versions["controller_states"].items():
        path = (ROOT / specification["path"]).resolve()
        weight_checks[state] = {
            "resolved_path": str(path),
            "sha256": sha256(path),
            "matches_versions": sha256(path) == specification["sha256"],
        }
    generator_path = (ROOT / versions["generator_path"]).resolve()
    weight_checks["generator"] = {
        "resolved_path": str(generator_path),
        "sha256": sha256(generator_path),
        "matches_versions": sha256(generator_path)
        == versions["generator_weights_sha256"],
    }
    snapshot_path = (
        ROOT
        / "results/round_resource_status_20261012/snapshot_1791782554748181514.json"
    )
    snapshot = read_json(snapshot_path)
    snapshot_rows = [
        row
        for row in snapshot["entries"]
        if row["directory"] == "controller_state_diagnosis_v2_20261012"
    ]
    resource = {
        "snapshot_sha256": sha256(snapshot_path),
        "status": snapshot["status"],
        "controller_rows_exact": snapshot_rows == [
            {"directory": "controller_state_diagnosis_v2_20261012", **row}
            for row in ledger
        ],
        "own_seconds": float(sum(row["seconds"] for row in ledger)),
        "own_seconds_within_600": sum(row["seconds"] for row in ledger) <= 600,
        "all_exit_zero": all(row["exit_code"] == 0 for row in ledger),
        "cap_checks": snapshot["cap_checks"],
        "charged_wall_hours_one_gpu_per_job": snapshot[
            "charged_wall_hours_one_gpu_per_job"
        ],
        "output_bytes": snapshot["output_bytes"],
        "data_free_bytes": snapshot["data_free_bytes"],
    }
    report = {
        "status": "PASS" if not (
            source_hash_mismatches
            or all_metric_differences
            or binding_failures
            or failure_differences
        ) else "FAIL",
        "method": "independent NumPy implementation; original metric module not imported",
        "numpy_version": np.__version__,
        "source_hashes": {
            "bound_files": len(versions["files"]),
            "mismatches": source_hash_mismatches,
        },
        "audited_hashes": audited_hashes,
        "manifest": manifest_summary,
        "reference": {
            "keys": reference_keys,
            "file_sha256": sha256(OUT / "references/fixed.npz"),
            "content_sha256": reference_content_hash,
            "file_hash_matches_preparation": sha256(OUT / "references/fixed.npz")
            == preparation["reference_sha256"],
            "content_hash_matches_requests": all(
                read_json(OUT / "requests" / f"{state}.json")["replay_motion"]
                == reference_content_hash
                for state in STATES
            ),
            "provenance_classification": "controller-derived synthetic proxy from a prior simulated trajectory; dataset-derived endpoint and heading targets",
            "provenance_checks": reference_provenance_checks,
        },
        "state_reports": state_reports,
        "all_metric_differences": all_metric_differences,
        "binding_failures": binding_failures,
        "cross_state": {
            "initial_archives_exact": initial_cross_state_exact,
            "metadata_equal": metadata_cross_state,
            "common_valid_reference_max_abs_difference": common_reference_max,
            "hidden_physx_identity_established": False,
        },
        "failure_components": {
            "differences": failure_differences,
            "input_hashes_match": failure_components["input_hashes"]
            == expected_failure_hashes,
            "source_hash_matches": failure_components["source_sha256"]
            == sha256(ROOT / "scripts/diagnose_controller_state_failures_20261012.py"),
            "new_simulator_batches": failure_components["new_simulator_batches"],
            "generation_calls": failure_components["generation_calls"],
            "thresholds_changed": failure_components["thresholds_changed"],
            "alternative_success_rule": failure_components[
                "alternative_success_rule"
            ],
        },
        "preservation": preservation,
        "terminal": terminal_checks,
        "weights": weight_checks,
        "upstream": {
            "commit": upstream_commit,
            "matches_versions": upstream_commit == versions["upstream_commit"],
            "tracked_closd_diff": upstream_diff,
        },
        "resource": resource,
        "normalization": {
            "aggregate_means_are_plain_arithmetic_means": True,
            "classification_thresholds_source": "frozen_thresholds.json",
            "prediction_statistic_denominator_used": False,
            "note": "rank_candidate threshold ratios are not called by this diagnosis and are not reported metrics",
        },
        "scope": {
            "states": 3,
            "same_algorithm_different_training_states": True,
            "sources": 24,
            "offsets": [-0.15, 0.0, 0.15],
            "rows_per_state": 72,
            "seed": 11,
            "families": ["jump", "dance"],
            "official_train_sources": manifest_summary["validation_official_splits"].get("train", 0),
            "official_test_sources": manifest_summary["validation_official_splits"].get("test", 0),
            "independent_final_test": False,
            "candidate_efficacy_test": False,
            "generation_calls": 0,
            "simulation_only": True,
            "tracking_reference": "fixed replay reference built from a prior simulated controller trajectory and used as a synthetic tracking proxy",
        },
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = main()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        Path(arguments.output).write_text(encoded + "\n", encoding="utf8")
    print(encoded)
