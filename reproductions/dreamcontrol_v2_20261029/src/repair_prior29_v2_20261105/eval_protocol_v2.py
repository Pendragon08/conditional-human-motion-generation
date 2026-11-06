"""CPU evaluation protocol adapter over hash-bound, precomputed embeddings.

This module never loads a diffusion/policy model or starts a GPU job. FID,
diversity and embedding distances come from the pinned upstream metrics module.
Geometry values must be produced by the same pinned upstream geometric metrics.
Four repeated task texts cannot be relabeled as 32-way paper R-Precision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

TASKS = ("pick", "pour", "bimanual_box_lift", "box_push_pull")
STAGES = ("raw_diffusion", "raw_reencoded", "postoptimized")
METRICS_SHA = "021108acce91e7a4e887ce9b8ff39a2628d9fbbfc54bbd765bee7e24218c1a77"
POOL_SHA = "5fc1a0beb071e7a65b68e689a57ed0ba09b633a83d5ec5fd261b1620115bb1cb"
ENCODER_FIXED = {
    "weights_sha256": "569bbd5526cb4b5fdda48a08abf7dd0b5a1e9a39ea0be381730771ac18063981",
    "mean_sha256": "0bdb5ba69a3a9e34d71990db15bc535ebc024c8d95ddb5574196f96058faa7d3",
    "std_sha256": "6a5f7d60301c9465972fc225f8ad0ee8f957e7720431189123eb6d15873a9557",
    "wrapper_sha256": "7a3578bae99c75c11ab741d12d95b3048f3d7f768b2508a7a54d2f5f5288c330",
    "feature_dim": 263, "embedding_dim": 512, "max_feature_frames": 196,
    "embedding_scale": "raw_encoder_output", "normalization": "published_humanml3d_mean_std",
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def normalized_caption(caption):
    return " ".join(caption.casefold().split())


def require_sha(value, label):
    if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
        raise ValueError("Invalid SHA binding: " + label)


def validate_inputs(manifest, arrays, geometry):
    if manifest.get("schema") != "generated_reference_eval_inputs_v2":
        raise ValueError("Unknown embedding-cache schema")
    if manifest.get("retrieval_group_definition") != "authorized_four_task_ids":
        raise ValueError("This four-task cache must use the four actual semantic task groups")
    encoder = manifest["encoder"]
    for key, value in ENCODER_FIXED.items():
        if encoder.get(key) != value:
            raise ValueError("Encoder/normalization/scale mismatch: " + key)
    encoder_hash = content_sha(encoder)
    if manifest["encoder_fingerprint_sha256"] != encoder_hash:
        raise ValueError("Encoder fingerprint mismatch")
    pool = manifest["reference_pool"]
    if (pool["name"], pool["split"], pool["count"], pool["source_manifest_sha256"]) != ("project586", "test", 586, POOL_SHA):
        raise ValueError("This protocol preserves the exact project586 held-out pool")
    if pool["encoder_fingerprint_sha256"] != encoder_hash:
        raise ValueError("Ground-truth encoder differs from generated encoder")
    rows = manifest["records"]
    ids = [r["id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Missing/repeated sample IDs")
    for row in rows:
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("Sample IDs must be nonempty strings")
        if row["task"] not in TASKS or not isinstance(row["admitted"], bool):
            raise ValueError("Invalid task/admission record")
        if not row["caption"].strip() or not row["retrieval_group_id"]:
            raise ValueError("Caption and explicit semantic retrieval group required")
        if row["retrieval_group_id"] != row["task"]:
            raise ValueError("Do not split one task into fabricated semantic retrieval groups")
        if row.get("caption_role") != "actual_generation_condition":
            raise ValueError("Evaluation caption must be the actual generation condition")
        require_sha(row["source_sample_sha256"], "generated sample")
        require_sha(row["initial_noise_sha256"], "initial noise")
        require_sha(row["feature_content_sha256"], "raw feature content")
    if len({r["initial_noise_sha256"] for r in rows}) != len(rows) or len({r["feature_content_sha256"] for r in rows}) != len(rows):
        raise ValueError("Repeated initial noise or raw feature content")
    if set(manifest["stages"]) != set(STAGES):
        raise ValueError("Keep raw, raw round-trip, and postoptimization separate")
    specs = [pool, manifest["matched_text"], *manifest["stages"].values()]
    if manifest.get("legacy_retrieval"):
        specs.append(manifest["legacy_retrieval"]["text_pool"])
    for spec in specs:
        a = arrays[spec["embedding_key"]]
        if a.ndim != 2 or a.shape[1] != 512 or a.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(a).all():
            raise ValueError("Embeddings must be finite unscaled floating N x 512 arrays")
        if spec["embedding_content_sha256"] != array_sha(a):
            raise ValueError("Embedding content SHA mismatch: " + spec["embedding_key"])
        if spec["encoder_fingerprint_sha256"] != encoder_hash:
            raise ValueError("Embedding encoder provenance mismatch")
    if len(arrays[pool["embedding_key"]]) != 586 or manifest["matched_text"]["ids"] != ids:
        raise ValueError("Ground-truth/text cache ordering mismatch")
    if len(arrays[manifest["matched_text"]["embedding_key"]]) != len(ids):
        raise ValueError("Matched-text count mismatch")
    by_id = {r["id"]: r for r in rows}
    for stage, spec in manifest["stages"].items():
        stage_ids = spec["ids"]
        if len(stage_ids) != len(set(stage_ids)) or not set(stage_ids) <= set(ids):
            raise ValueError("Stage contains duplicate or unknown IDs: " + stage)
        if len(arrays[spec["embedding_key"]]) != len(stage_ids):
            raise ValueError("Stage embedding count mismatch")
        if len(spec["source_artifact_sha256"]) != len(stage_ids):
            raise ValueError("Each stage row needs an artifact source binding")
        for h in spec["source_artifact_sha256"]:
            require_sha(h, stage)
        if stage in ("raw_diffusion", "raw_reencoded") and set(stage_ids) != set(ids):
            raise ValueError("Raw and round-trip metrics must cover all generated candidates")
        if not {r["id"] for r in rows if r["admitted"]} <= set(stage_ids):
            raise ValueError("Admitted rows are missing a stage")
    legacy = manifest.get("legacy_retrieval")
    if legacy:
        captions = legacy["text_pool"]["captions"]
        if len(captions) != len(arrays[legacy["text_pool"]["embedding_key"]]):
            raise ValueError("Legacy text-pool caption count mismatch")
        if len({normalized_caption(c) for c in captions}) != len(captions):
            raise ValueError("Legacy text pool repeats normalized captions")
        for plan in legacy["plans"]:
            if plan["id"] not in by_id:
                raise ValueError("Legacy plan has an unknown sample ID")
            try:
                caption = captions[plan["text_indices"][plan["positive_column"]]]
            except (IndexError, TypeError):
                raise ValueError("Legacy positive label is outside the text pool") from None
            if normalized_caption(caption) != normalized_caption(by_id[plan["id"]]["caption"]):
                raise ValueError("Legacy positive text is not the generation condition")
    if geometry.get("schema") != "generated_reference_geometry_cache_v2" or geometry["upstream_metrics_sha256"] != METRICS_SHA:
        raise ValueError("Geometry cache must bind the original metric functions")
    if geometry["generation_manifest_sha256"] != manifest["generation_manifest_sha256"]:
        raise ValueError("Geometry and embedding generation manifests differ")
    geo = geometry["records"]
    if len({r["id"] for r in geo}) != len(geo) or set(r["id"] for r in geo) != set(ids):
        raise ValueError("Geometry sample IDs differ")
    for row in geo:
        if row["source_sample_sha256"] != by_id[row["id"]]["source_sample_sha256"]:
            raise ValueError("Geometry/sample source SHA mismatch")
        for stage, values in row["stages"].items():
            if stage not in STAGES:
                raise ValueError("Unknown geometry stage")
            if values["control_length_unit"] != "m" or values["skating_scale"] != "fraction_0_1":
                raise ValueError("Geometry unit/scale mismatch")
            if not np.isfinite(values["control_l2_m"]) or values["control_l2_m"] < 0 or not 0 <= values["skating_ratio"] <= 1:
                raise ValueError("Invalid geometric metric value")
            spec = manifest["stages"][stage]
            if row["id"] not in spec["ids"]:
                raise ValueError("Geometry has a stage without its embedding")
            expected = spec["source_artifact_sha256"][spec["ids"].index(row["id"])]
            if values["source_artifact_sha256"] != expected:
                raise ValueError("Geometry/stage artifact SHA mismatch")
        if not {s for s, spec in manifest["stages"].items() if row["id"] in spec["ids"]} <= set(row["stages"]):
            raise ValueError("Geometry stage missing for an available embedding")
    return by_id, {r["id"]: r for r in geo}


def build_paired_batch_plan(records, batch_size=32, seed=20261105):
    """Each motion and conditioning text stay paired; no same-group negatives."""
    if batch_size != 32:
        raise ValueError("Paper-style protocol uses exactly 32 paired candidates")
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate retrieval IDs")
    queues = defaultdict(list)
    caption_group = {}
    for row in records:
        caption = normalized_caption(row["caption"])
        group = row["retrieval_group_id"]
        if caption in caption_group and caption_group[caption] != group:
            raise ValueError("Same text assigned different semantic groups")
        caption_group[caption] = group
        queues[group].append(row["id"])
    rng = np.random.default_rng(seed)
    for group in sorted(queues):
        rng.shuffle(queues[group])
    batches = []
    while True:
        active = sorted(group for group, queue in queues.items() if queue)
        if len(active) < batch_size:
            break
        chosen = rng.choice(len(active), batch_size, replace=False)
        batch = [queues[active[int(i)]].pop() for i in chosen]
        batches.append(batch)
    unused = sorted(item for queue in queues.values() for item in queue)
    return {"schema": "paired_unique_semantic_group_batch32_v2", "seed": seed,
            "batch_size": 32, "semantic_groups": len(queues), "batches": batches,
            "evaluated_samples": sum(map(len, batches)), "excluded_incomplete_batch_ids": unused,
            "status": "ready" if batches else "not_applicable_fewer_than_32_semantic_groups",
            "paper_fullmix_comparable": False}


def rank_scores(distance, positive):
    """Stable rank plus explicit tie counts; no silent optimistic tie breaking."""
    order = np.argsort(distance, axis=-1, kind="stable")
    rank = np.array([int(np.where(order[i] == positive[i])[0][0]) + 1 for i in range(len(order))])
    targets = distance[np.arange(len(distance)), positive]
    ties = np.sum(np.isclose(distance, targets[:, None], atol=1e-12, rtol=0), axis=1) - 1
    return rank, ties


def score_paired_batches(plan, motion_by_id, text_by_id, distance_function):
    if plan["status"] != "ready":
        return {"status": plan["status"], "r_precision_32": None,
                "semantic_groups": plan["semantic_groups"], "reason": "Four fixed skill prompts do not supply 32 distinct paired semantic labels."}
    ranks, ties = [], []
    for ids in plan["batches"]:
        distances = distance_function(np.stack([motion_by_id[x] for x in ids]), np.stack([text_by_id[x] for x in ids]))
        r, tie = rank_scores(distances, np.arange(32))
        ranks.extend(r.tolist())
        ties.extend(tie.tolist())
    return {"status": "computed", "evaluated_samples": len(ranks),
            "r_precision_32": {"top_" + str(k): float(np.mean(np.asarray(ranks) <= k)) for k in (1, 2, 3)},
            "queries_with_tied_positive_distance": int(np.count_nonzero(ties)),
            "plan_sha256": content_sha(plan), "paper_fullmix_comparable": False}


def score_four_group_diagnostic(records, motion_by_id, text_by_id, distance_function):
    groups = defaultdict(list)
    for row in records:
        groups[row["retrieval_group_id"]].append(row)
    if len(groups) != 4:
        return {"status": "not_applicable_requires_exactly_four_fixed_text_groups", "r_at_4_diagnostic": None}
    names = sorted(groups)
    texts = []
    for name in names:
        rows = groups[name]
        captions = {normalized_caption(r["caption"]) for r in rows}
        first = text_by_id[rows[0]["id"]]
        if len(captions) != 1 or any(not np.allclose(text_by_id[r["id"]], first, atol=1e-6, rtol=0) for r in rows):
            return {"status": "not_applicable_group_contains_different_text_embeddings", "r_at_4_diagnostic": None}
        texts.append(first)
    ids = [r["id"] for r in records]
    target = np.array([names.index(r["retrieval_group_id"]) for r in records])
    ranks, ties = rank_scores(distance_function(np.stack([motion_by_id[x] for x in ids]), np.stack(texts)), target)
    return {"status": "computed", "diagnostic": "four_fixed_task_text_group_retrieval",
            "r_at_4_diagnostic": {"top_" + str(k): float(np.mean(ranks <= k)) for k in (1, 2, 3)},
            "balanced_random_reference": {"top_1": .25, "top_2": .50, "top_3": .75},
            "group_names": names, "queries_with_tied_positive_distance": int(np.count_nonzero(ties)),
            "paper_r_precision": False}


def score_legacy_plan(legacy, motion_by_id, text_array, distance_function):
    """Reuse saved exact 1+31 plans. Never silently choose a new distractor pool."""
    if not legacy:
        return {"status": "missing_exact_legacy_plan", "legacy_r_precision_32": None}
    if content_sha(legacy["plans"]) != legacy["plans_sha256"]:
        raise ValueError("Legacy candidate plan SHA mismatch")
    seen, ranks, ties = set(), [], []
    for row in legacy["plans"]:
        if row["id"] not in motion_by_id:
            continue
        if row["id"] in seen:
            raise ValueError("Repeated legacy query ID")
        seen.add(row["id"])
        indices = row["text_indices"]
        if len(indices) != 32 or len(set(indices)) != 32 or not 0 <= row["positive_column"] < 32:
            raise ValueError("Legacy plan must contain 32 distinct text indices")
        if min(indices) < 0 or max(indices) >= len(text_array):
            raise ValueError("Legacy text index outside cache")
        d = distance_function(motion_by_id[row["id"]][None], text_array[indices])
        rank, tie = rank_scores(d, [row["positive_column"]])
        ranks.extend(rank.tolist())
        ties.extend(tie.tolist())
    if seen != set(motion_by_id):
        raise ValueError("Legacy plan does not cover evaluated sample IDs")
    return {"status": "computed", "legacy_r_precision_32": {"top_" + str(k): float(np.mean(np.asarray(ranks) <= k)) for k in (1, 2, 3)},
            "evaluated_samples": len(ranks), "plans_sha256": legacy["plans_sha256"],
            "queries_with_tied_positive_distance": int(np.count_nonzero(ties)),
            "scope": "legacy fixed task condition plus 31 held-out weak-label distractors; not paired batch retrieval",
            "paper_r_precision": False}


def summarize_geometry(ids, geometry, stage):
    values = [geometry[x]["stages"][stage] for x in ids]
    return {key: {"mean": float(np.mean([v[key] for v in values])),
                  "median": float(np.median([v[key] for v in values])),
                  "p95": float(np.quantile([v[key] for v in values], .95))}
            for key in ("control_l2_m", "skating_ratio")}


def evaluate(manifest, arrays, geometry, metrics, seed=20261105):
    by_id, geo = validate_inputs(manifest, arrays, geometry)
    rows = manifest["records"]
    texts = {row["id"]: value for row, value in zip(rows, arrays[manifest["matched_text"]["embedding_key"]])}
    reference = arrays[manifest["reference_pool"]["embedding_key"]]
    gt_stats = metrics.calculate_activation_statistics(reference)
    def diversity(values):
        if len(values) <= 32:
            return {"status": "insufficient_samples_for_original_32_pair_function", "mean": None}
        old_state = np.random.get_state()
        try:
            draws = []
            for i in range(20):
                np.random.seed(seed + i)
                draws.append(float(metrics.calculate_diversity(values, 32)))
            return {"status": "computed", "mean": float(np.mean(draws)), "sd_across_pair_draws": float(np.std(draws)), "pairs_per_draw": 32, "draws": 20}
        finally:
            np.random.set_state(old_state)
    report = {"schema": "generated_reference_metrics_protocol_v2", "status": "completed",
              "reference_pool": manifest["reference_pool"], "encoder": manifest["encoder"],
              "seed": seed, "upstream_metrics_sha256": METRICS_SHA,
              "paper_fullmix_comparable": False, "ground_truth_diversity": diversity(reference),
              "stage_definitions": {"raw_diffusion": "unmodified denormalized generator 263D output",
                                    "raw_reencoded": "the same generated points passed through the final-reference encoder",
                                    "postoptimized": "final robot FK points passed through that same encoder"},
              "populations": {}, "retrieval_plans": {},
              "metric_directions": {"FID": "lower", "R-Precision": "higher within identical valid protocol",
                                    "Diversity": "closer to ground-truth", "Control L2": "lower", "Skating Ratio": "lower"}}
    populations = {"all_generated_candidates": rows, "admitted_references": [r for r in rows if r["admitted"]]}
    for population, population_rows in populations.items():
        population_ids = {r["id"] for r in population_rows}
        result = {"population_count": len(population_rows), "tasks": {}}
        for task in (*TASKS, "all"):
            task_rows = [r for r in population_rows if task == "all" or r["task"] == task]
            task_ids = {r["id"] for r in task_rows}
            task_report = {"population_count": len(task_rows), "stages": {}}
            for stage in STAGES:
                spec = manifest["stages"][stage]
                motion = {i: value for i, value in zip(spec["ids"], arrays[spec["embedding_key"]]) if i in task_ids}
                evaluated_rows = [r for r in task_rows if r["id"] in motion]
                evaluated_ids = [r["id"] for r in evaluated_rows]
                if not evaluated_rows:
                    task_report["stages"][stage] = {"status": "no_available_stage", "evaluated_count": 0}
                    continue
                values = np.stack([motion[i] for i in evaluated_ids])
                fid = None
                if len(values) > 1:
                    fid = float(metrics.calculate_frechet_distance(*gt_stats, *metrics.calculate_activation_statistics(values)))
                    if not np.isfinite(fid) or fid < -1e-6:
                        raise ValueError("Invalid upstream FID value")
                plan = build_paired_batch_plan(evaluated_rows, seed=seed)
                plan_key = population + "/" + task + "/" + stage
                report["retrieval_plans"][plan_key] = plan
                legacy = manifest.get("legacy_retrieval")
                legacy_text = arrays[legacy["text_pool"]["embedding_key"]] if legacy else None
                task_report["stages"][stage] = {
                    "status": "computed", "evaluated_count": len(evaluated_rows),
                    "population_coverage": len(evaluated_rows) / max(1, len(task_rows)),
                    "evaluated_ids_sha256": content_sha(evaluated_ids),
                    "fid_to_project586": fid, "diversity": diversity(values),
                    "geometry": summarize_geometry(evaluated_ids, geo, stage),
                    "paired_batch32_retrieval": score_paired_batches(plan, motion, texts, metrics.euclidean_distance_matrix),
                    "four_group_diagnostic": score_four_group_diagnostic(evaluated_rows, motion, texts, metrics.euclidean_distance_matrix),
                    "legacy_project_retrieval": score_legacy_plan(legacy, motion, legacy_text, metrics.euclidean_distance_matrix)}
            result["tasks"][task] = task_report
        report["populations"][population] = result
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "embeddings", "geometry", "metrics-source"):
        p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    args = p.parse_args()
    bindings = {}
    for name in ("manifest", "embeddings", "geometry", "metrics_source"):
        path, expected = getattr(args, name), getattr(args, name + "_sha256")
        if sha_file(path) != expected:
            raise ValueError("Input file SHA mismatch: " + name)
        bindings[name] = {"path": str(path), "sha256": expected}
    if args.metrics_source_sha256 != METRICS_SHA:
        raise ValueError("A different upstream metric implementation requires a new protocol version")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    with np.load(args.embeddings, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    spec = importlib.util.spec_from_file_location("pinned_dreamcontrol_metrics_v2", args.metrics_source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    report = evaluate(manifest, arrays, geometry, metrics, seed=args.seed)
    report["input_bindings"] = bindings
    report["protocol_source_sha256"] = sha_file(__file__)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "output": str(args.output), "sha256": sha_file(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
