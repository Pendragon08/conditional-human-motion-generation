# Experiment Audit Report — condition feedback probe V2

**Date:** 2026-10-11  
**Auditor:** fresh Codex reviewer, read-only, same-family  
**Acceptance status:** provisional  
**Remote target:** `/data/code/ouyanghao/oyh/conditional-human-motion-generation` at wrapper-repository HEAD `e1ef719d6702d15a5ab211db0e55be438b2d1c88`  
**Audit mode:** direct source/artifact reads and read-only recomputation with the recorded Python environment; no collector/GPU rerun, installation, remote write, licensed-array download, or Overleaf action

## Overall verdict: WARN

The saved V2 probe is internally consistent and its numeric summaries are reproducible. I found no fabricated result, self-normalized score, missing batch, broken request/output binding, category flip hidden by tolerance, or uncharged retry.

The warning is about what the probe can establish. It is a four-source, one-generation-seed simulator diagnostic. Its tracking target is a selected generated/simulated reference, and the physical slip proxy is not reproducible across the preserved V1/V2 same-arm `r0` repeats even though their kinematics are bit-identical. Therefore this integrity pass must not be interpreted as efficacy, physical-validity, real-world, semantic, or novelty evidence. The saved summary itself correctly reports a physical floor and withholds those claims.

**Blocking issues:** none for retaining and reporting this probe as an exploratory diagnostic. The unsupported physical-proxy validity, zero physical-pass rate, missing semantic verification, one-seed scope, and experiment-internal split block any efficacy or novelty claim.

## A. Ground-truth and reference provenance: WARN

The fixed task goals are not generated from the evaluated model output. The collector reconstructs dataset motion in the simulator frame and freezes the endpoint-derived goal before generation (`scripts/physics_guided_collect.py:94-106`). The evaluator reads `metadata['original_goal']`, not the proposed/generated goal, as the target (`scripts/physics_guided_core.py:100-118`), and computes condition gates against that fixed target (`scripts/physics_guided_core.py:142-153`). Metadata explicitly labels the goal as dataset-derived and fixed before generation (`results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r0/metadata.json:685`).

The manifest retains cache paths, original source paths, captions, and official split metadata (`results/condition_feedback_probe_v2_20261011/source_manifest.json:5-22`, `:25-42`, `:45-62`, `:65-82`). Direct verification found the same case-object hash, `80f8b15d94c746681f042b9e1dd18b4c512844f8a8ba84d2f3ea1184ef2e73d6`, and original-goal hash, `74b8ca3ee31aa96c4a1271dd38510f3cd0f1b573b2e09c4758ce8ec43d6de70b`, in all 13 batches. The generator, controller, evaluator, motion inputs, semantic inputs, source manifest, and audited implementation files all match the recorded SHA-256 values (`results/condition_feedback_probe_v2_20261011/versions.json:3-37`).

Two qualifications require visible labeling:

- `split: "val"` is an experiment-internal designation. The four rows comprise two `official_split: "train"` and two `official_split: "test"` sources (`source_manifest.json:14,21,34,41,54,61,74,81`). It must not be presented as an official validation split.
- Replay tracking uses a selected physics-refined generated/simulated trajectory. Selection is explicit (`scripts/physics_guided_queue.py:136-169`), and the document says the selected reference is independently replayed (`docs/condition_feedback_probe_v2_20261011.md:19-23`). This component is a `synthetic_proxy`, not dataset ground truth.

No fake-ground-truth failure was found; the WARN records the mixed provenance and naming risk.

## B. Score normalization: PASS

The reported endpoint, heading, timing, tracking, penetration, slip, and jerk values are raw SI-unit measurements or direct fractions (`scripts/physics_guided_core.py:100-155`). Candidate ranking divides errors only by fixed, predeclared thresholds (`scripts/physics_guided_core.py:158-165`), not by maxima, minima, or means of the model's own output. Dataset mean/std normalization occurs only on model input features (`scripts/physics_guided_collect.py:56-62`) and is not metric normalization.

The frozen thresholds match both the current configuration and the historical frozen-threshold file exactly (`results/condition_feedback_probe_v2_20261011/frozen_thresholds.json:2-20`; `results/condition_feedback_probe_v2_20261011/versions.json:46`). Every execution request binds the same threshold-file hash; for example `results/condition_feedback_probe_v2_20261011/requests/val_proposed_s11_r1.json:7-10`.

## C. Result existence, recomputation, and lineage: PASS

All expected artifacts exist: 13 requests, 13 raw directories, 13 receipts, 13 metadata files, 13 metric files, 13 lineage files, 13 numeric certificates, 13 initial-state archives, and 13 trajectory archives. The state is complete (`results/condition_feedback_probe_v2_20261011/state.json:2-3`). The ledger contains exactly 13 unique successful charges (`results/condition_feedback_probe_v2_20261011/budget_ledger.json:2-79`).

For every raw batch, I independently verified:

- request equals the execution-bound lineage request;
- `metadata.json`, `metrics.json`, and `trajectories.npz` hashes equal the lineage outputs;
- the receipt trajectory hash equals the saved trajectory file;
- exactly 12 cases and 12 metric rows are present;
- reset/cache gates are true and every exported reset-vs-first-action difference is zero;
- `reference_frame == arange(120)`;
- the metric rows recompute from the saved arrays under the frozen thresholds.

A representative construction binding is at `results/condition_feedback_probe_v2_20261011/raw/val_proposed_s11_r1/lineage.json:2-26`; a representative replay binding is at `results/condition_feedback_probe_v2_20261011/raw/val_eval_proposed_s11/lineage.json:2-21`.

### Numeric equivalence and adversarial negative tests

Across all 13 metric files, independent recomputation found 36 nonzero floating differences. Every difference is exactly `4.440892098500626e-16`; the allowed bound is `7.105427357601002e-15`. No type, key, list length, identifier, counter, `None`, string, or classification changed. The comparator requires exact types/keys and exact non-float values (`scripts/metric_numeric_equivalence.py:9-31`) and applies tolerance only to finite floats (`scripts/metric_numeric_equivalence.py:19-26`).

The preserved V1 failure was reproduced directly from its saved arrays. `metrics[1].heading_error_rad` is saved as `0.07121248986374962` (`results/condition_feedback_probe_20261011/raw/val_native_s11_r1/metrics.json:70-71`) and recomputes as `0.07121248986375006`, a difference of `4.440892098500626e-16`. The V2 certificate records the same values and tolerance (`results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r1/metric_numerics.json:7-18`).

Independent in-memory negative tests confirmed that the comparator rejects:

- a `joint_pass` Boolean flip;
- a `1e-6` endpoint-error change;
- a `1e-6` threshold mutation through the exact threshold-file hash binding.

Thus tolerance did not accept a category flip or threshold change. Saved scores remained unchanged.

### Summary reproduction

The four arm summaries and all three pairwise difference objects reproduce exactly from the saved replay metric rows and selected construction status. The saved result reports 12 attempts per arm, zero physical/joint passes, and the exact endpoint reductions (`results/condition_feedback_probe_v2_20261011/probe_summary.json:3-85`). It also truthfully reports `numeric_pilot_gate: false`, `novelty: not_established`, `confirmatory: false`, four sources, one seed, a physical-quality floor, and unverified semantics (`probe_summary.json:87-105`).

## D. Dead metric code: PASS

The metric implementation is live. The collector calls `metrics(...)` to create each saved row (`scripts/physics_guided_collect.py:274-288`), and the cache verifier calls it again for numeric certification (`scripts/physics_guided_numeric_lineage.py:82-103`). `surface_metrics` feeds the physical fields (`scripts/physics_guided_core.py:70-97,139-149`), and `rank_candidate` feeds candidate selection (`scripts/physics_guided_queue.py:136-151`). No defined evaluation metric in the audited files was found to be dead or absent from results.

## E. Scope, pairing, attempts, failures, and budget: WARN

The saved claims accurately disclose the limited scope: four preselected sources, three offsets, one generation seed, three candidates per arm, 108 construction attempts, and 48 independent replay attempts (`docs/condition_feedback_probe_v2_20261011.md:17-23`; `probe_summary.json:90-100`). This is not the planned 432-input/three-seed test.

Pairing and arm construction passed direct checks:

- Six proposed/no-feedback/shuffled follow-up batches bind `noise_round: 0` (`requests/val_proposed_s11_r1.json:3-15` and corresponding requests), and their metadata carries the same generation-noise hash as the common `val_native_s11_r0` batch (`raw/val_native_s11_r0/metadata.json:1084`; `raw/val_proposed_s11_r1/metadata.json:1084`; `raw/val_no_feedback_s11_r1/metadata.json:1084`). Native random candidates `r1/r2` use distinct noise rounds.
- All six feedback payloads independently recompute from their stated parent metrics. Proposed feedback is samplewise; shuffled feedback maps each recipient to another source in the same family at the same offset; no-feedback zeros both position and heading feedback. The implementation is at `scripts/physics_guided_queue.py:105-133`; saved flags and parent paths are at `results/condition_feedback_probe_v2_20261011/feedback/*.json:93-95`.
- The no-feedback batches are real completed computations: the ledger charges them (`budget_ledger.json:33-42`), their metadata records about 30 seconds and one generator batch call (`raw/val_no_feedback_s11_r1/metadata.json:1087-1088`), and their receipt records 12 completed cases (`raw/val_no_feedback_s11_r1/receipt.json:2-7`). Their trajectory semantic hashes differ across `r1/r2`, so they are not file copies.
- Every arm has three real raw candidate batches. The saved selector chose the correct argmin for all 48 choices, and selected-motion, selection-JSON, and all candidate-output hashes match (`selected/proposed_s11.lineage.json:2-20`). Each replay request binds the selected semantic motion hash (`requests/val_eval_proposed_s11.json:3-11`), and all four replay references match the selected canonical reference with maximum error `0.0`.

All 108 V2 construction cases happened to complete. The denominator behavior is structurally correct but not empirically exercised by an incomplete construction in V2: analysis forces a selected incomplete construction to `joint_pass = false` and still counts all 12 rows (`scripts/run_condition_feedback_probe_v2_20261011.py:109-123`).

The historical failure and cost are retained. V1 remains `FAILED_RETAINED` (`results/condition_feedback_probe_20261011/state.json:2-4`) with two successful charged batches (`results/condition_feedback_probe_20261011/budget_ledger.json:2-13`), totaling `65.8949820175767` seconds. V2 records `427.6622204128653` seconds, matching `charged_gpu_hours` exactly (`probe_summary.json:106`). The 216-attempt reset repair also remains bound (`results/reset_alignment_v4_20261011/measurement_repair_summary.json:3-6`).

### Physical proxy validity is not supported

The preserved V1 and V2 `val_native_s11_r0` batches use identical initial states, original goal, environment seed, generation noise, generation prefix, adapter source, reference arrays, and actual kinematics. `actual`, `reference`, position/orientation, linear/angular velocity, validity, termination, and frame indices are bit-identical. However, their contact-force arrays differ by as much as `744.8496704101562 N`.

That difference changes support observations and slip metrics despite identical motion. For case 0, V1 reports slip mean `0.0971128071`, slip fraction `0.3542600897`, and 223 support observations (`results/condition_feedback_probe_20261011/raw/val_native_s11_r0/metrics.json:32-34`); V2 reports `0.0845158821`, `0.3192488263`, and 213 (`results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r0/metrics.json:32-34`). Support counts differ in 10/12 cases and both slip values differ in 12/12. The Boolean physical classification does not flip because every case is already a failure, but that does not validate the continuous proxy.

The saved summary is appropriately conservative: all arms have `physical_pass: 0`, `physical_quality_floor: true`, and “Physical proxy discrimination remains unresolved” (`probe_summary.json:8,22,36,50,99-102`). This physical proxy cannot presently support an efficacy conclusion.

## F. Evaluation-type classification: WARN

- **Overall:** `simulation_only`.
- **Condition target provenance:** dataset-derived fixed endpoint plus predeclared offsets; not model-output ground truth.
- **Tracking component:** `synthetic_proxy`, because the replay target is a selected generated/simulated trajectory.
- **Physical component:** `simulation_only` proxy based on simulator contact force and collision geometry (`scripts/physics_guided_core.py:70-97`). Its repeat validity is unsupported as described above.
- **Semantic component:** not evaluated; metadata says independent generated semantics are not verified (`raw/val_native_s11_r0/metadata.json:688`), and the summary says `semantics_independently_verified: false` (`probe_summary.json:102`).
- **Human evaluation / real robot:** absent.

## Claim impact

- **Artifact provenance, request/output binding, and saved numeric summaries:** supported for this exact snapshot.
- **Descriptive condition-error reductions in this four-source, one-seed simulator probe:** supported, with the saved qualification.
- **Physical-quality superiority or validated physics guidance:** unsupported; proxy repeat validity is not established and all physical gates fail.
- **Semantic preservation, diversity, general efficacy, robustness, real-world transfer, or novelty:** unsupported.
- **Universal ineffectiveness of the mechanism:** unsupported; the saved report correctly avoids this claim.

## Required claim boundary

Any downstream report should call this a validation-only exploratory simulator diagnostic, state that `val` is an experiment-internal split over two official-train and two official-test clips, identify replay tracking as a synthetic proxy, report the physical-proxy repeat discrepancy, and retain the zero physical/joint pass result. A confirmatory efficacy claim remains blocked until the physical observation/proxy is validated and the deferred independent scope and semantic guards are completed.

