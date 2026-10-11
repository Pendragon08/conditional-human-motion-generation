# Reset-Alignment V3 Experiment Audit

**Date:** 2026-10-11  
**Auditor:** `/root/physics_guided_integrity_auditor` (fresh Codex reviewer, read-only)  
**Review class:** same-family, provisional  
**Result family:** `results/reset_alignment_v3_20261011` only  
**Overall verdict:** **FAIL — partial execution and failed initialization-identity gate**  
**Record integrity:** **PASS WITH WARNINGS**  
**Scientific validity:** **FAIL / no controlled comparison available**

V3 is a preserved failed measurement attempt, not a completed 216-case study. The documented launcher ran exactly once with no retry. Its first child batch, `planned_s11`, completed 12 of 12 cases, 120 frames each, with no termination. Before the second batch launched, the outer process stopped at the GPU resource guard and exited 1. The remaining seven batches and 204 planned attempts never started.

The V3 horizontal registration succeeds for the one completed batch: actual versus reference root XY is below 1 mm, the reference translation preserves relative pose, displacement, and height to float32 precision, and the four sources' requested -0.15/0/+0.15 m target offsets reproduce within 3.064e-6 m in absolute world coordinates. The stricter initialization gate fails. Root state, DOF positions, and DOF velocities are identical between the bootstrap reset capture and the first action, but all four rigid-body tensor diagnostics differ, with maxima from 0.0350 to 6.5191 in their native units. Since the raw initial seven tensors were not saved, those differences and paired hashes can be verified as recorded but cannot be recomputed from raw arrays or attributed to a specific reset/synchronization cause.

The correct stop is therefore `STOP_MEASUREMENT_INVALID`. The 0/12 joint result is not evidence that the mechanism is ineffective, and this family must not be pooled with the historical run.

## Execution and stop verification

- The fresh witness records one invocation from `2026-10-11T11:24:42.3527247Z` to `11:25:34.3809353Z`, outer exit code 1, and zero retries.
- Exact outer stdout is `RESET_ALIGNMENT_BATCH planned_s11 0`, confirming the only child batch exited 0.
- Exact stderr points to `scripts/run_reset_alignment_checks_v3.py:89`: `AssertionError: No displacement of another workload`.
- The job list contains eight batches (`scripts/run_reset_alignment_checks_v3.py:78-80`). The GPU guard runs before request creation and child launch (`:81-100`), so absence of a second request proves that the later seven children did not start.
- Failure-time GPU memory/utilization values were not printed. The guard failure is verified, but which predicate failed cannot be determined. The witness preflight was 699 MiB and 0% utilization; that earlier sample does not describe the later failure sample.
- `state.json` still says `RUNNING/planned_s11`. This is stale: the launcher sets it before a child launch at `scripts/run_reset_alignment_checks_v3.py:93` and only writes a retained-failure state after a nonzero child exit at `:111-113`. The inter-batch guard assertion bypasses that branch. The outer exit record is final, and a later read-only process check found no V3 launcher or collector process.

## Actual coverage

| Item | Planned | Actually observed |
|---|---:|---:|
| Batches | 8 | 1 completed child batch |
| Attempts | 216 | 12 |
| Generation seeds | 11, 12, 13 | 11 only |
| Modes | planned, native | planned only |
| Full validation batches | 2 x 72 | 0 |
| Fixture sources | 4 per batch | 4 |
| Offsets per source | -0.15, 0, +0.15 m | all three in the first batch |

The completed fixture has 12/12 completed, 0 terminated, 4 condition passes, 0 physical passes, 12 tracking passes, and 0 joint passes. Cross-seed, cross-mode, noise-distinctness, shared-prefix, and full-validation claims are not covered.

## Initialization and coordinate checks

The V3 adapter resets before anchoring, saves the seven exported tensor views, then uniformly translates the dataset reference before constructing the immutable target (`scripts/reset_aligned_collect_v3.py:13-24,25-41`). At the first action it records XY error, tensor differences/hashes, reset count, motion IDs, time offsets, and global offsets (`:55-70`). The environment seed is fixed at 11 while the requested generation seed remains in metadata (`:42-54,73-74`).

First-batch results:

| Check | Result |
|---|---:|
| Actual/reference root XY maximum component | 0.000712544 m |
| Actual/reference root XY maximum Euclidean error | 0.000718999 m |
| Bootstrap reset/reference root XY error | 0 m |
| Reference relative-pose preservation error | 5.96046e-8 m |
| Reference frame-displacement preservation error | 5.96046e-8 m |
| Reference height change | 0 m |
| Maximum absolute +/-0.15 m goal-offset error | 3.06368e-6 m |
| Maximum offset error after subtracting each reset root | 3.07336e-7 m |
| Maximum reset-root difference within a source triplet | 2.86102e-6 m |

The offset and XY checks pass for the first fixture. They cannot certify cross-seed or cross-mode stability because those batches do not exist.

The initialization identity diagnostics fail:

| Exported tensor | Recorded maximum difference | Paired SHA-256 equal |
|---|---:|---:|
| `_humanoid_root_states` | 0 | yes |
| `_dof_pos` | 0 | yes |
| `_dof_vel` | 0 | yes |
| `_rigid_body_pos` | 0.0350069702 | no |
| `_rigid_body_rot` | 1.4754348993 | no |
| `_rigid_body_vel` | 3.7104039192 | no |
| `_rigid_body_ang_vel` | 6.5190958977 | no |

The V3 full analyzer requires every recorded tensor difference to be below 1e-5 and every paired hash to match (`scripts/analyze_reset_alignment_v3.py:23-37,78-80`). The retained batch fails that declared criterion. These seven exported tensor views do not include contacts, PhysX hidden state, RNG state, or policy observations; no complete simulator-state identity is claimed.

## A. Ground-truth provenance: PASS WITH QUALIFICATION

The measurement manifest contains four preselected HumanML validation sources, two jump and two dance, chosen by manifest order before execution (`scripts/run_reset_alignment_checks_v3.py:34-42`). The reference motion and endpoint targets are dataset-derived. Generated semantic quality is explicitly not verified by an independent evaluator in the retained metrics. This is a coordinate/initialization diagnostic, not semantic ground truth evaluation.

## B. Own-output normalization: PASS

The request binds the frozen threshold file SHA-256 `451fae0a...`; the collector uses the unchanged historical metric code. No score is divided by a statistic of the evaluated output. `quality_thresholds_changed=false` in the derived summary is consistent with the request and version records.

## C. Claimed versus existing results: PASS WITH WARNINGS

- Exactly one execution request and one result directory exist. Receipt, metrics, metadata, trajectory, and lineage files are present for `planned_s11`; there are no result or request files for the seven unstarted batches.
- The execution-bound request is byte-for-byte equal to the request embedded in lineage. The versions-file hash, manifest hash, and all three lineage output hashes match. Certification is `bound_request_at_execution` and `confirmatory=false`.
- The NPZ hash matches the receipt and lineage. It contains 1,440 valid frames, reference indices exactly 0 through 119, zero terminations, and finite numeric arrays.
- All 13 version-listed project files still match their recorded hashes according to the read-only witness. The request's frozen source-config and threshold hashes match their recorded versions.
- `measurement_repair_summary.json` correctly reports 12/216, one started batch, seven named unstarted batches, `coordinate_gate_pass=false`, no cross-seed/mode checks, no full-validation quality, and `STOP_MEASUREMENT_INVALID`.
- The stale `RUNNING` state is preserved and explicitly explained in the summary. It must never be used as evidence that the job is active or successful.

## D. Dead metric code: PASS WITH QUALIFICATION

The collector and historical metric functions produced all 12 retained rows. The normal complete-run analyzer was not invoked because its first assertion requires collection completion (`scripts/analyze_reset_alignment_v3.py:9-12`). The CPU-only incomplete-run analyzer instead reads the retained files, checks bindings and offsets, records missing coverage, and emits the failure summary (`scripts/analyze_incomplete_reset_measurement.py:13-96`). No output from the unexecuted complete-run analyzer is claimed.

## E. Scope, seeds, failures, and resource limits: FAIL FOR SCIENTIFIC SCOPE

Only 5.56% of planned attempts and one of eight batches exist. One seed and one mode cannot test cross-seed or cross-mode comparability. The full validation comparison is absent. The initialization identity gate fails before any efficacy interpretation.

Resource accounting is within limits but required one correction. Child ledgers total 7,139.618935 s or 1.983227482 h when combined with the historical/V1/V2 records. The V3 outer invocation lasted 52.0282106 s while its child ledger recorded 31.99840576 s. Charging the unrecorded 20.02980484 s wrapper overhead gives the conservative combined total **7,159.648740 s = 1.988791317 h**. The updated derived summary now records this conservative value and its witness hash. It remains far below the 72 h total limit and 4 h measurement limit. Related result directories use about 1.86 GiB and the witness observed 40.63 GiB free, satisfying the 15 GiB output and 20 GiB reserve limits.

`gpu_hours` here is conservative charged wall time, not measured GPU kernel occupancy.

## F. Evaluation type: simulation-only diagnostic with real dataset conditions

- **real_gt:** HumanML source motion/text and dataset-derived endpoint conditions.
- **simulation_only:** trajectories, reset state, contact proxies, and success metrics are collected in Isaac Gym.
- **synthetic_proxy:** generated motions and physical proxy metrics.
- **human_eval:** none.
- **real_robot:** none.

## Derived summary audit

Current `measurement_repair_summary.json` SHA-256: `09bfaaa424d643eaa8ad64d42145f0a6156ddd722e5d90d2f92a6a8587e9c146`.

Independent CPU recomputation confirms:

- `planned_attempts=216`, `attempts=12`, `completed_cases=12`, `started_batches=1`;
- the exact seven unstarted batch names;
- the 4/0/12/0 condition/physical/tracking/joint counts;
- request equality and all lineage output hashes;
- every reported offset error;
- the failed seven-tensor identity gate;
- conservative combined charged time `1.9887913167262192 h`, including outer overhead;
- `next_stage=STOP_MEASUREMENT_INVALID`.

The summary's stop conclusion is supported. The resource assertion's sampled values remain unavailable, so the audit does not identify which GPU bound failed.

## Claim impact

- **“V3 completed 216 attempts.” — false; 12 attempts completed.**
- **“V3 demonstrated cross-seed or cross-mode coordinate consistency.” — unsupported; those batches did not start.**
- **“The first-batch XY anchor and +/-15 cm offsets are aligned.” — supported for the four retained sources only.**
- **“The bootstrap and first-action initialization states are identical.” — contradicted by four rigid-body tensor diagnostics.**
- **“Joint pass 0/12 shows the mechanism is ineffective.” — unsupported.** The batch is initialization-invalid, incomplete, and contains one seed/mode.
- **“The resource guard stopped before displacing another workload.” — supported in the limited sense that the next child was never launched.** Failure-time GPU values were not logged.

## Required handling

1. Preserve this directory as a failed, non-confirmatory V3 attempt; do not rerun into it or rewrite its execution records.
2. Do not pool its 12 rows with the historical run or use 0/12 as an efficacy or inefficacy result.
3. If another repair is attempted, use a new result family and save the raw before/after initialization tensors, relevant simulator-state synchronization points, and the failure-time GPU sample.
4. Keep the conservative outer-wall resource charge of 1.988791317 h in cumulative reporting.

## Evidence hashes

- `planned_s11/metadata.json`: `14dd190dd8d7c4f1aa4309ca20705c1390abc07dbe590ac28a9b9840268fef5d`
- `planned_s11/metrics.json`: `3075c125ef53ec6357896c5c23e45566914d4ed61e287bde31794915cd1be0aa`
- `planned_s11/trajectories.npz`: `453d0d2c4f344e42c71e493a9dad13a4f8cbd5576d701723108898ec9c0b90cc`
- `planned_s11/lineage.json`: `9fa6f67a6e0087e0c3cabb8910747b7e31ca0198b510b786fd11017c6db007d7`
- `planned_s11_execution_request.json`: `9d75184600cdf6670cac22da0d4fae305c65a1f8e661eb1f7ada7ad137b80a5e`
- `versions_before_execution.json`: `f819cb4a67c298dcb0210a2f8bbf0359b171cd952ef1525bb3273a0f6054a758`
- `state.json`: `5ee65da98d0376962a69d0b2b8c240fda304feedd75a96e23f7ddb2e53ea3cba`
- `measurement_repair_summary.json`: `09bfaaa424d643eaa8ad64d42145f0a6156ddd722e5d90d2f92a6a8587e9c146`
- independent witness JSON: `c84c5338d442a3661ae12681ad49734e5162097436799eaebb5a0dbb1b6e97ff`
- independent recomputation JSON: `b69e4e3390d37559b67cb2db2537a43c95c3d006864267c37cda7c897d18199d`
- recomputation script: `15d198ebbce3c30f4794d28cac651e0797aad484058ae5ed089874eafa2b67c9`

This audit ran no GPU workload and performed no retry. It did not modify the remote result directory or any execution code/output. Only audit records and read-only evidence copies were created locally.
