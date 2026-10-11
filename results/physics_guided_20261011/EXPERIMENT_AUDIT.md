# Physics-Guided Experiment Audit Report

**Date:** 2026-10-11  
**Auditor:** `/root/physics_guided_integrity_auditor` (fresh Codex reviewer, read-only)  
**Review class:** same-family, provisional  
**Project:** `conditional-human-motion-generation`  
**Audited result family:** historical run `results/physics_guided_20261011` only  
**Overall verdict:** **WARN**  
**Record integrity:** **PASS WITH WARNINGS**  
**Scientific validity of efficacy, mechanism, seed, and offset conclusions:** **FAIL / unsupported**

The historical result package is unusually complete and candid. Every published count and gate value that I recomputed matches its source data, all 74 registered batches exist and hash correctly, resource accounting is within the stated limits, and the report explicitly says that G1 and downstream training were not run. I found no fabricated result, phantom G1 result, self-normalized score, or hidden failed gate.

The old experiment nevertheless cannot support efficacy or mechanism conclusions. The collector builds each reference trajectory and goal from the simulator state present after actor creation, while later episode resets place the controlled character at a different world anchor. The error is large: reference frame 0 versus the actual post-step pelvis differs by a median of 0.80–0.87 m and by as much as 1.65 m. A separate 288-record offset recomputation finds a maximum intended-versus-realized world-offset error of 2.53584 m. The joint gate is also at a near-total floor because the slip proxy rejects almost every completed attempt. These defects make the all-zero gain and confidence intervals descriptive outputs of this amended diagnostic run, not evidence that the proposed mechanism works or does not work.

Later reset-alignment V1, V2, or V3 scripts and results are outside this audit. They cannot retroactively validate the historical run; any repaired measurement must be judged as a new result family.

## Critical finding: reference and actual states use different world anchors

`scripts/physics_guided_collect.py:87-105` constructs `dataset_sim`, `reference`, and the endpoint goal from `_rigid_body_pos` before the evaluated episode reset. The upstream environment assigns random actor-creation XY positions in `third_party/CLoSD/closd/env/tasks/humanoid_im.py:911-924`; later episode resets follow the fixed reset path in `third_party/CLoSD/closd/env/tasks/humanoid_im.py:900-968` and `third_party/CLoSD/closd/env/tasks/humanoid.py:504-519,552-562`. The reference and actual episode therefore do not receive the same world translation.

Independent recomputation from saved arrays gives:

| Seed | Median reference-frame0/actual XY error | Maximum error |
|---:|---:|---:|
| 11 | 0.865609 m | 1.593060 m |
| 12 | 0.799619 m | 1.591299 m |
| 13 | 0.816911 m | 1.653108 m |

Cross-seed world goals differ substantially: seed 12 versus 11 has a 1.978951 m maximum absolute difference and seed 13 versus 11 has a 1.883929 m maximum. Removing each source's reference-frame0 pelvis makes the source-relative goal and prefix nearly equal (roughly 1.6e-5 m), but this does not repair evaluation geometry because the actual reset state did not receive the same translation. Goal relative to the actual initial state changes by up to 2.429034 m for seed 12 and 2.078818 m for seed 13. Thus the world difference is not validly described as a harmless global translation.

Across 288 nonzero-offset records, the maximum norm between the requested +/-0.15 m offset and the realized world goal difference is 2.535842 m; the corresponding maximum after expressing the goal relative to the actual initial state is 2.535205 m. The detailed per-case evidence is in `historical_coordinate_per_case.json`, reproduced by `recompute_historical_coordinates.py`.

Within a fixed `(case, seed)` pair, every compared method uses the same saved original goal and case order. That preserves descriptive paired counts within that pair. It does not make seeds or nominal offsets controlled replicates, and it does not rescue causal or mechanism interpretation.

## A. Ground-truth provenance: PASS WITH QUALIFICATION

- `scripts/physics_guided_prepare.py:31-48` loads cached HumanML data and links clips to dataset indices; `scripts/physics_guided_prepare.py:46-66` retains source path and text provenance.
- `scripts/physics_guided_prepare.py:70-97` groups by original `source_path` and constructs a deterministic internal train/validation/test split. I independently counted 200 unique source groups: 128 train, 24 validation, and 48 test, with zero pairwise source overlap. Each of the three motion families contributes 64/12/24 groups.
- The 48 study-test sources include 41 items whose official HumanML origin is train and 7 whose origin is official test. The internal source-group split is sound for this study, but it does not establish independence from the pretrained generator or policy.
- The condition target comes from the dataset-derived reference endpoint plus a requested offset (`scripts/physics_guided_collect.py:102-106`). The evaluated motion is the simulated actual state, and the semantic metric is a frozen learned proxy over those states (`scripts/physics_guided_semantics.py:3,33-36,54-57,87-122`).
- This passes the anti-fake-ground-truth check: the target is not manufactured from the evaluated model's output. The coordinate-anchor defect above invalidates the physical comparison even though provenance is genuine.

## B. Own-output normalization: PASS

- The SI-unit thresholds are fixed in `configs/physics_guided_20261011.json:17-32` and frozen in `frozen_thresholds.json`.
- Candidate ranking in `scripts/physics_guided_core.py:158-165` divides errors by fixed thresholds; it never divides by the current prediction's maximum, minimum, or mean.
- Raw metric values and pass indicators are both retained. No audited metric uses output-dependent normalization, and no suspicious near-one score is manufactured by renormalization.

## C. Claimed versus existing results: WARN

- Every required historical summary artifact exists: `summary.json`, `progression_gate.json`, `semantic_diversity_gate.json`, `complete_attempt_register.json`, `lineage_register.json`, `cross_seed_coordinate_checks.json`, `resource_summary.json`, `failure_case_catalog.json`, `per_family_summary.json`, `publication_versions.json`, `experiment_report.html`, `overleaf_experiment_record.tex`, and `all_construction_and_replay_attempts.csv`.
- Independent recomputation from `per_attempt.csv` reproduces every published arm count, point estimate, per-seed joint-pass rate, comparison gain, confidence interval, and numeric-gate Boolean exactly. There are 10,512 construction/replay CSV rows, 74 registered complete labels, and 4,320 main replay attempts.
- A read-only hash scan of all 74 registered `metadata.json`, `metrics.json`, and trajectory files found zero register or lineage mismatches.
- The report does not claim G1 or downstream training: `results/physics_guided_20261011/experiment_report.html:8,58`; `results/physics_guided_20261011/overleaf_experiment_record.tex:10,41,47,54`. `progression_gate.json` records `g1=not_run`, `downstream_training=not_run`, and `novelty=not_established`.
- Lineage is mixed but disclosed: 61 batches were request-bound at execution and 13 were certified retrospectively by exact-array and metric replay. Every lineage record is marked `confirmatory=false`; retrospective certification is compatibility evidence, not contemporaneous request provenance.
- Minor publication mismatch: the saved HTML snapshot still uses a stale title containing “受控诊断实验” at `experiment_report.html:2`, although its body clearly disclaims controlled inference. The current publisher source changes the title at `scripts/publish_physics_guided_final.py:203`; the already-published HTML has not been regenerated from that source. The audited `publication_versions.json` correctly binds the older publisher hash, so this is a stale artifact, not a concealed rewrite.

## D. Dead metric code: PASS

- The core witness and measurement checks are invoked in `scripts/physics_guided_run.sh:18-23`.
- The collector is called by the check, collection, and queue paths in `scripts/physics_guided_run.sh:22-30`.
- The main analyzer and semantic evaluator are invoked by `scripts/physics_guided_queue.py:262-280`.
- Outputs from the condition, physical, tracking, penetration, slip, semantic, and joint metrics appear in the batch metrics and published summary/component files. I found no main reported metric implemented only as dead code.

## E. Scope, seeds, failures, and statistical support: WARN

- Main scope: 10 arms, 48 test clips, 3 diffusion seeds (11, 12, 13), and 432 attempts per arm. Six equal-budget three-candidate arms enter the primary numeric gate: `random_best`, `native_best`, `no_feedback`, `shuffled`, `planned_prefix`, and `tracking_only` (`scripts/physics_guided_analyze.py:74-82`; queue construction in `scripts/physics_guided_queue.py:229-260`).
- The analysis uses a source-clustered, family-stratified bootstrap (`scripts/physics_guided_analyze.py:55-73`). That is structurally appropriate for the planned unit of resampling.
- The run records six development/witness batches, 30 known extra attempts, one early failed batch with unknown attempt count, five CPU queue failures, eight amendment records, and nine version snapshots. These are retained rather than silently discarded.
- Scope does not repair the coordinate confound: diffusion seed also changes the reference world anchor, and nominal offset does not equal realized goal-relative-to-actual displacement.
- The gate has a severe floor. Validation already had zero joint passes, yet thresholds were frozen before validation-distribution calibration; the protocol discloses this deviation at `docs/physics_guided_20261011.md:25`. Across main arms, penetration passes 432/432, slip passes only 1-5/432, physical passes only 1-2/432, and all equal-budget comparisons have 0.0 percentage-point gain with a `[0.0, 0.0]` interval. A degenerate conjunction cannot distinguish methods.

## F. Evaluation type: WARN — mixed

- **real_gt:** HumanML source motion/text and endpoint condition provenance.
- **synthetic_proxy:** model-produced candidate motions and their replayed simulation traces.
- **simulation_only:** all physical success, completion, contact-force, penetration, and slip measurements occur in Isaac Gym simulation.
- **learned proxy:** the semantic/diversity evaluation embeds simulated actual-state motion using frozen HumanML evaluator assets.
- **human evaluation:** none.
- **real-robot evaluation:** none.

The semantic gate is correctly labeled as a proxy. It cannot substitute for independent semantic acceptance of the generation reference, and it does not make the physical gate valid.

## Additional integrity controls

| Control | Status | Evidence and limitation |
|---|---|---|
| Source-group split | PASS | `physics_guided_prepare.py:70-97`; 128/24/48 groups, zero overlap. Pretrained-weight independence is not established. |
| Immutable targets | PASS within case/seed; FAIL across seeds/offset geometry | Original target/hash checks in `physics_guided_collect.py:109,114,126`; paired methods share targets. World anchors and realized offset geometry differ as quantified above. |
| Actual/reference timestamps | PASS structurally, WARN semantically | Explicit `reference_frame`; action is applied before physics and state recorded after the step (`physics_guided_collect.py:190-243`); saved indices must be 0..119 (`:278-279`). Old data lacks a pre-action state and begins from the anchor mismatch. |
| Condition-update bounds | PASS | Total translation <=0.15 m, per-update translation <=0.05 m, heading step <=5 degrees and total <=15 degrees (`physics_guided_collect.py:109-118`). |
| Candidate-budget fairness | PASS for declared primary gate | Six primary comparison arms receive the same three-candidate budget. One-candidate raw/ordinary/native-single arms are excluded from the equal-budget numeric gate. |
| Slip proxy geometry | WARN | Exact collision primitives are parsed (`physics_guided_core.py:40-67`); point velocity uses `v + omega x r`, force/height define support, and surface geometry defines penetration (`:70-97`). Body-level net force does not localize contact to the selected surface point, and the resulting proxy is at a floor. |
| Gate logic | PASS structurally; FAIL inferentially | All equal-budget arms, fixed thresholds, semantic asset hashes, and no invented downstream stage are checked (`physics_guided_analyze.py:74-123`). The inputs do not satisfy coordinate comparability and the joint endpoint is degenerate. |
| Resource limits | PASS | GPU time 1.958902242 h <72 h; output 1,997,173,593 bytes <15 GiB; free space 43,630,018,560 bytes >20 GiB. |

## Independently reproduced results

| Arm | Attempts | Completed | Construction-completed | Joint pass | Joint rate |
|---|---:|---:|---:|---:|---:|
| raw | 432 | 279 | 276 | 1 | 0.0023148 |
| ordinary | 432 | 263 | 276 | 0 | 0 |
| random_best | 432 | 265 | 287 | 0 | 0 |
| native_single | 432 | 280 | 298 | 0 | 0 |
| native_best | 432 | 279 | 313 | 0 | 0 |
| proposed | 432 | 276 | 302 | 0 | 0 |
| no_feedback | 432 | 280 | 298 | 0 | 0 |
| shuffled | 432 | 280 | 303 | 0 | 0 |
| planned_prefix | 432 | 260 | 280 | 0 | 0 |
| tracking_only | 432 | 280 | 302 | 0 | 0 |

All seven published comparisons against `proposed` have 0.0 percentage-point joint-rate gain, a `[0.0, 0.0]` clustered interval, and zero per-seed gain. The primary numeric gate is false for all six equal-budget arms.

The physical-component recomputation shows the floor directly:

| Arm | Physical pass | Slip pass | Penetration pass | Median completed slip fraction |
|---|---:|---:|---:|---:|
| raw | 2/432 | 2/432 | 432/432 | 0.734694 |
| ordinary | 1/432 | 1/432 | 432/432 | 0.744186 |
| random_best | 1/432 | 3/432 | 432/432 | 0.748603 |
| native_single | 1/432 | 2/432 | 432/432 | 0.755756 |
| native_best | 1/432 | 2/432 | 432/432 | 0.750000 |
| proposed | 1/432 | 2/432 | 432/432 | 0.754937 |
| no_feedback | 2/432 | 4/432 | 432/432 | 0.757453 |
| shuffled | 2/432 | 5/432 | 432/432 | 0.759987 |
| planned_prefix | 1/432 | 2/432 | 432/432 | 0.747478 |
| tracking_only | 1/432 | 3/432 | 432/432 | 0.750000 |

Semantic evaluation contains 30 arm-by-seed entries, 4,320 attempts, and 2,743 completed traces. For the disclosed proxy, `proposed` versus `random_best` has matching 6.6184 versus 6.7064, diversity 5.2923 versus 5.29525, and multimodality 4.38723 versus 4.29587. Evaluator assets and input groups are hash-bound, but these are diagnostic learned-proxy values.

## Claim impact

- **“The historical package faithfully reports what was run.” — supported with a minor publication-version warning.** Counts, hashes, statuses, failures, resource use, and gate outputs reproduce.
- **“The proposed method improves the joint physical-condition-tracking endpoint.” — unsupported.** The numeric gate is false, the coordinate frame is invalid, and the endpoint is at a floor.
- **“The proposed mechanism is ineffective.” — unsupported.** An invalid and degenerate measurement cannot establish absence of an effect.
- **“Results are robust across three seeds or +/-0.15 m offsets.” — unsupported.** Seeds and nominal offsets are confounded with meter-scale world-anchor changes.
- **“Semantic quality/diversity was evaluated.” — supported only as a frozen actual-state learned proxy.** It is not independent semantic ground truth.
- **“G1 or downstream training was completed.” — not claimed and not present.** The record correctly marks both stages not run.

## Required actions before scientific use

1. Do not reuse the historical numbers as evidence for efficacy, inefficacy, robustness, seed effects, or offset effects. Retain them only as an amended diagnostic record.
2. Any repaired run must place reference, target, and actual reset state in one frozen coordinate frame before the first action, save the pre-action actual state, verify the mapping per case, and use a new result directory and lineage chain.
3. Calibrate and freeze a physically meaningful slip/contact threshold on a separate validation set. Report component outcomes as well as the joint conjunction so a floor cannot masquerade as equivalence.
4. Preserve all execution-time requests and retrospective amendments. Keep retrospective compatibility records explicitly non-confirmatory.
5. Regenerate the public HTML title from the current publisher if the stale “controlled” title remains visible, while preserving the historical producer hash and prior artifact.

## Audit artifacts and hashes

- `input_hashes.json` — SHA-256 `eccebfb3670a948ae713a76b2660ba225bbb25409b52d952d0f1f560f469e9f4`
- `independent_recomputation.json` — SHA-256 `3a7c7eec8113a29b16d440cdc4cba7f551fd42df629c0e96d12b2f45736c3ce0`
- `recompute_published_results.py` — SHA-256 `56f28744e338cca59f18518867bb1122ada5138aace0982b424e56f77e0837fa`
- `historical_coordinate_per_case.json` — SHA-256 `4f5765132e39d0176c81b50ab5ad7c8e9996a1723d07ef45af55cdfe490964ca`
- `recompute_historical_coordinates.py` — SHA-256 `3d4bb5467c7be0a51c9cdc0296217b1393804c7c7ba378eaa85a3210d3a4f327`

This audit was read-only with respect to experiment code and outputs. It ran no GPU experiment and changed no historical result. The only files written are the independent audit trace artifacts under this audit directory.
