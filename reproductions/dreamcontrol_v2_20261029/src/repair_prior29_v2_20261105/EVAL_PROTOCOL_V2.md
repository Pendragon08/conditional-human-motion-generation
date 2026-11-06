# Generated-reference evaluation protocol v2

`eval_protocol_v2.py` implements the fourth audit correction as executable input validation, population/stage separation, retrieval planning, and scoring. It consumes hash-bound caches created by the existing upstream matching networks and geometric functions. It does not run a model, use a GPU, regenerate references, modify old results, or replace Full-Mix with project data under the paper's name.

## Three separately reported retrieval quantities

1. **Legacy project R@32** retains the previous project's one fixed task instruction plus 31 weak-label held-out distractors. The cache producer must preserve the exact text pool, query-to-candidate indices, and positive column in a SHA-bound plan. The adapter refuses silent resampling, incorrect positive captions, missing query IDs, or changed plan hashes. This remains a project diagnostic and is compared only under the same plan/text conditions; changed generation captions must be disclosed and do not create a strict paired old/new comparison automatically.
2. **Paired batch R@32** uses 32 real motion–conditioning-text pairs, with 32 distinct semantic groups, and keeps each matching pair together. Duplicate text or same-group paraphrases cannot become false negatives. The supplied four-task cache is required to use the actual four task IDs as its semantic groups. Consequently the current 2,000 references produce **N/A for paired batch R@32**, with a machine-readable reason and no numerical substitute. The generic batch planner is implemented and tested with genuinely distinct synthetic groups; it does not authorize generating a new benchmark.
3. **R@4 task-text diagnostic** compares each motion embedding to the four actual fixed task-text embeddings. It is explicitly named `r_at_4_diagnostic`, never paper R-Precision. With balanced groups, random Top-1/2/3 reference values are 25/50/75%. It is computed only if each group has one actual conditioning text and consistent text embeddings. Per-task reports have fewer than four groups and therefore show N/A; the combined four-task report contains R@4.

Even a valid batch R@32 would not make the project586 dataset numerically comparable to the paper Full-Mix benchmark. `paper_fullmix_comparable` remains false. Exact distance ties are counted and reported, not hidden.

## Population and processing stages

The report separates `all_generated_candidates` from `admitted_references`, and reports Pick, Pour, bimanual box lift, push/pull, and their union. It preserves all generated raw and round-trip rows, including rejected candidates. Every stage records its actual available sample count, population coverage, and evaluated-ID hash. A rejected candidate without a final robot trajectory is not treated as a completed postoptimized row.

Three stages have separate embedding arrays:

| Stage key | Input to the existing matching encoder |
| --- | --- |
| `raw_diffusion` | Original, denormalized 263D diffusion output |
| `raw_reencoded` | The same raw decoded points processed by the same feature encoder used for final FK points |
| `postoptimized` | Saved final robot FK points processed by that same feature encoder |

For the admitted population, all three stages must contain every admitted ID, providing an actual within-candidate comparison. The unchanged project586 held-out pool and original matching-network mean/std are used throughout. FID is measured on unscaled encoder embeddings; Diversity reuses the public 32-pair function with the previous 20 deterministic draws. Fewer than 33 available clips give an explicit unavailable Diversity value instead of changing that function's protocol. Higher Diversity is not automatically better: compare to the same ground-truth pool.

Control L2 and Skating Ratio must come from the original pinned geometric functions. Control L2 is in metres; skating is a fraction in [0,1]. Round-trip reencoding changes the embedding representation, not the physical trajectory, so its geometry values should be the same raw points/targets as `raw_diffusion`. The cache must bind each stage's source artifact SHA. These generation metrics never substitute for downstream task success or actual interaction-point error.

## Cache schema and upstream bindings

The executable entry point accepts a JSON manifest, a NumPy NPZ embedding cache, a JSON geometry cache, and the unchanged original `metrics.py`. All four file SHA256 values are mandatory. Array-content hashes use contiguous NumPy bytes, preserving their recorded float32 or float64 dtype. Each array is N×512 and must share the same encoder fingerprint.

The exact project586 source-manifest SHA is `5fc1a0beb071e7a65b68e689a57ed0ba09b633a83d5ec5fd261b1620115bb1cb`. The pinned upstream metrics SHA is `021108acce91e7a4e887ce9b8ff39a2628d9fbbfc54bbd765bee7e24218c1a77`. `ENCODER_FIXED` in the adapter specifies the already-used evaluator weights, mean, std, wrapper hashes, 263D input, 512D output, 196-feature-frame cap, and raw-embedding scale. Normalizing or rescaling embeddings, changing the ground-truth pool, or passing centimetres/percentages is rejected.

Minimal manifest structure (replace all placeholders with actual values):

```json
{
  "schema": "generated_reference_eval_inputs_v2",
  "retrieval_group_definition": "authorized_four_task_ids",
  "generation_manifest_sha256": "SHA_OF_NEW_GENERATION_MANIFEST",
  "encoder": "COPY_OF_ENCODER_FIXED_MAPPING",
  "encoder_fingerprint_sha256": "CONTENT_SHA_OF_ENCODER_MAPPING",
  "reference_pool": {
    "name": "project586", "split": "test", "count": 586,
    "source_manifest_sha256": "5fc1a0beb071e7a65b68e689a57ed0ba09b633a83d5ec5fd261b1620115bb1cb",
    "embedding_key": "reference", "embedding_content_sha256": "ARRAY_SHA",
    "encoder_fingerprint_sha256": "ENCODER_FINGERPRINT"
  },
  "records": [{
    "id": "pick:NEW_ID", "task": "pick", "admitted": true,
    "caption": "ACTUAL_GENERATION_CONDITION", "caption_role": "actual_generation_condition",
    "retrieval_group_id": "pick", "source_sample_sha256": "SAMPLE_SHA",
    "initial_noise_sha256": "NOISE_SHA", "feature_content_sha256": "FEATURE_SHA"
  }],
  "matched_text": {
    "embedding_key": "text", "ids": ["pick:NEW_ID"],
    "embedding_content_sha256": "ARRAY_SHA", "encoder_fingerprint_sha256": "ENCODER_FINGERPRINT"
  },
  "stages": {
    "raw_diffusion": {
      "embedding_key": "raw", "ids": ["pick:NEW_ID"], "source_artifact_sha256": ["SAMPLE_SHA"],
      "embedding_content_sha256": "ARRAY_SHA", "encoder_fingerprint_sha256": "ENCODER_FINGERPRINT"
    },
    "raw_reencoded": "SAME_SPEC_STRUCTURE_WITH_ROUNDTRIP_ARRAY_AND_SOURCE_HASHES",
    "postoptimized": "SAME_SPEC_STRUCTURE_WITH_AVAILABLE_FITTED_IDS_AND_ROBOT29_HASHES"
  }
}
```

The cache producer can import `content_sha`, `array_sha`, `ENCODER_FIXED`, `STAGES`, and `TASKS` from the adapter. `matched_text.ids` must exactly equal manifest record order. Stage arrays have their own explicit ID order. Stage source hashes bind the actual generated/round-trip/fitted artifacts; the producer must verify those artifacts before encoding. The adapter checks the supplied binding consistency but does not reread full trajectories itself.

The geometry JSON has schema `generated_reference_geometry_cache_v2`, the same `generation_manifest_sha256`, `upstream_metrics_sha256`, and `records`. Each record contains `id`, `source_sample_sha256`, and a `stages` mapping. Each available stage contains `source_artifact_sha256`, `control_l2_m`, `control_length_unit: "m"`, `skating_ratio`, and `skating_scale: "fraction_0_1"`.

Optional `legacy_retrieval` contains `text_pool` (the same embedding spec plus ordered `captions`), `plans` (`id`, 32 `text_indices`, and `positive_column`), and `plans_sha256 = content_sha(plans)`. If this exact cache is absent, the adapter reports `missing_exact_legacy_plan`; it does not invent an old-comparable value.

The old `motion_embeddings.npz` alone is insufficient: it contains two stages of admitted samples and lacks the reference embedding array, matched-text array, and raw round-trip stage. The root evaluator must form the new cache with the same existing upstream networks; do not relabel an old final array as round-trip data. Old completed reports remain the baseline evidence.

## Execution

```console
python src/repair_prior29_v2_20261105/eval_protocol_v2.py --manifest CACHE_MANIFEST.json --manifest-sha256 MANIFEST_SHA --embeddings EMBEDDINGS.npz --embeddings-sha256 EMBEDDINGS_SHA --geometry GEOMETRY.json --geometry-sha256 GEOMETRY_SHA --metrics-source ORIGINAL_METRICS.py --metrics-source-sha256 021108acce91e7a4e887ce9b8ff39a2628d9fbbfc54bbd765bee7e24218c1a77 --output NEW_REPORT.json
```

The output file must not exist. Nine CPU synthetic tests check 4-versus-32 grouping, false group splits, paired sample identities, source/array hashes, units/scales, fixed legacy plans, and stage/population coverage. They passed with existing local NumPy. A small metric stub in the population test checks orchestration only and is never used by the executable or reported as an experiment result. Actual new-reference metrics remain pending real generated artifacts and their upstream embedding/geometry caches.
