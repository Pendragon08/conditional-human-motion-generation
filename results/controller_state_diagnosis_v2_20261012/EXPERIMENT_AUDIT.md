# Experiment Audit Report

**Date:** 2026-10-12  
**Auditor:** fresh same-family Codex reviewer, read-only, provisional  
**Project:** controller-state diagnosis V2  
**Review independence:** same-family  
**Acceptance status:** provisional

## Overall Verdict: WARN

The recorded artifacts are internally consistent. An independent CPU implementation recomputed all 216 rows directly from the server-only trajectory archives without importing the experiment metric module. It reproduced every metric, Boolean classification, aggregate count, aggregate mean, denominator, criterion-failure count, and relevant hash with no differences. The actual replay and checkpoint bindings, visible reset invariants, budget ledger, preserved failures, and supplemental publication binding check also agree.

The WARN is about interpretation and remaining provenance limits, not a numerical mismatch. Tracking is measured against a controller-derived synthetic replay reference; `endpoint_time_error_s` is a closest-approach timing proxy rather than first target entry or deadline success; and the experiment compares one seed over 24 internal sources across three training states of the same algorithm. The final parent process has an external exit-0/sentinel observation but no automatic server-generated parent receipt.

The supported experimental conclusion is negative and diagnostic: **all three states have 0/72 joint passes** under this frozen setup. The numerical differences show controller-state sensitivity within this setup, but they do not establish an independently superior algorithm, robustness, generalization, innovation efficacy, semantic validity, or robot utility.

## Deterministic verification

- Independent implementation: NumPy 1.23.5, CPU only; original `physics_guided_metric_v23_20261012.py` was not imported.
- Rows recomputed: 216 of 216.
- Per-row metric or classification differences: 0.
- Aggregate count, mean, or denominator differences: 0.
- Input/checkpoint/reference binding failures: 0.
- Frozen criterion-breakdown differences: 0.
- Bound source mismatches: 0 of 47.
- Supplemental publication-binding guard: independently rerun read-only on the server, exit 0, all server-only raw bytes checked.
- Full evidence: `.aris/traces/controller_state_diagnosis_v2_20261012/integrity_audit/independent_recompute.json:1-961` and `publication_guard_recheck_stdout.json` in the same trace directory.

## Verified state results

All six reported numeric means have finite/total denominators of 72/72 for every state. Incomplete executions remain represented; endpoint values for those cases use their last valid state, while `joint_pass` still requires completion (`scripts/physics_guided_metric_v23_20261012.py:103-120,158-160`).

| State | Completed | Condition | Physical proxy | Tracking proxy | Joint |
|---|---:|---:|---:|---:|---:|
| multitask | 72 | 5 | 0 | 68 | 0 |
| t2m | 70 | 6 | 2 | 70 | 0 |
| unfinetuned | 71 | 7 | 2 | 70 | 0 |

Source: `results/controller_state_diagnosis_v2_20261012/analysis.json:4-10,59-65,114-120`.

| State | Endpoint error (m) | Heading error (rad) | Slip proxy fraction | Penetration fraction | Jerk p95 (m/s³) | Tracking mean (m) |
|---|---:|---:|---:|---:|---:|---:|
| multitask | 0.1959069926 | 0.2121799644 | 0.5424111457 | 0 | 1611.0568736 | 0.0550110892 |
| t2m | 0.1884172780 | 0.1706922124 | 0.3801818346 | 0 | 1336.1418139 | 0.0442855463 |
| unfinetuned | 0.1926381733 | 0.1548735931 | 0.3385630458 | 0 | 1198.3600121 | 0.0439587889 |

Source: `analysis.json:12-44,67-99,122-154`. The stored roundoff-equivalence records contain no changed values or classifications (`analysis.json:46-56,101-111,156-166`), and the independent implementation reached the same result.

The descriptive frozen-criterion decomposition also matches exactly:

| State | Position fails | Heading fails | Time-proxy fails | Penetration-proxy fails | Slip-proxy fails | Continuity-proxy fails | Tracking-mean fails | Tracking-p95 fails | Completion fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multitask | 57 | 25 | 40 | 0 | 72 | 61 | 4 | 4 | 0 |
| t2m | 57 | 18 | 37 | 0 | 70 | 54 | 2 | 2 | 2 |
| unfinetuned | 56 | 12 | 39 | 0 | 70 | 48 | 2 | 2 | 1 |

These are overlapping counts, not mutually exclusive causes. The script reads the frozen rows and thresholds, writes with exclusive creation, launches no simulation or generation, changes no threshold, and defines no alternative success rule (`scripts/diagnose_controller_state_failures_20261012.py:10-44`; `results/controller_state_diagnosis_v2_20261012/failure_components.json:2-111`). The output is numerically certified, but its claimed one-time command invocation has no separate terminal transcript.

## A. Ground Truth Provenance: WARN

Endpoint position and heading are derived from cached HumanML3D motion after the representation and simulator-coordinate conversion, then frozen before replay (`scripts/physics_guided_collect.py:57-62,94-105,129-138`). V23 scores these original targets, not a controller-adjusted target (`scripts/physics_guided_metric_v23_20261012.py:108-123`).

Tracking has a different provenance. The diagnosis copies `support_reference_probe_v63_20261012/references/baseline.npz` to `references/fixed.npz` (`scripts/controller_state_diagnosis_20261012.py:35-40`). Its producer fills that baseline from the `actual` trajectory of an earlier simulation and pads any invalid tail with the last valid pose (`scripts/prepare_support_reference_probe_v63_20261012.py:19-29,63`). The audit independently reconstructed the reference from that prior raw trajectory and obtained exact array equality; the two NPZ files also have the same SHA-256 (`.aris/traces/.../integrity_audit/independent_recompute.json:93-109`).

Therefore:

- Endpoint and heading targets: dataset-derived.
- Tracking reference: controller-derived synthetic proxy from an earlier simulated trajectory.
- Physical measures: simulated surface/contact/continuity proxies.
- Classification: `simulation_only + synthetic_proxy tracking`, not real tracking ground truth or robot truth.

The proxy status is disclosed in `versions.json:56-58`, `docs/controller_state_diagnosis_20261012.md:29`, and the report builder at `scripts/build_contact_measurement_report_20261011.py:515`. This avoids a fake-GT FAIL, but the controller-derived reference can favor behavior close to the incumbent trajectory and cannot support a real-GT tracking claim.

## B. Score Normalization: PASS

No output metric is divided by a maximum, minimum, or mean computed from the evaluated prediction. The only mean/std operation is an input representation transform (`scripts/physics_guided_collect.py:57-62`). The evaluator reports raw SI-unit/proxy values and applies frozen thresholds (`scripts/physics_guided_metric_v23_20261012.py:119-160`). State summaries use ordinary arithmetic means over finite values and publish their denominators (`scripts/controller_state_diagnosis_20261012.py:137-142`).

`rank_candidate` uses threshold ratios (`scripts/physics_guided_metric_v23_20261012.py:164-171`), but it is not called by this diagnosis and contributes no published result. The slip fraction divides by detected support observations (`metric_v23:87-99`), whose per-case counts vary (62–180 multitask, 34–201 t2m, 37–200 unfinetuned). That is an endogenous metric denominator worth showing alongside cross-state comparisons, but it is neither self-max normalization nor hidden score inflation. No case had zero support observations.

## C. Result Existence and Numeric Integrity: PASS

All claimed result files, keys, and rows exist. The audit verified raw trajectories, metadata, metrics, initial states, requests, lineages, receipts, loaded-input records, loaded-checkpoint records, source hashes, thresholds, reference content, checkpoint bytes, and summaries. The supplemental server guard independently checked the complete public binding and all server-only NPZ hashes (`scripts/controller_state_publication_guard_20261012.py:34-80`; `PUBLICATION_BINDING_SERVER.json:1-82`).

Actual loaded hashes match the frozen versions:

- generator: `3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618`
- multitask: `7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac`
- t2m: `62577dd2d127888108224d50b7a566eca0fdb650553d8564e268f72a80a2da84`
- unfinetuned: `a47bc0716ce0c40ffdbf1b093952e07182d47bf645123852df05c8c91faa68c2`
- upstream commit: `d8950bdb78312fcd2844f336b3fc079e7368c87f`

The source verifies and intercepts the bytes actually supplied to `torch.load` (`scripts/controller_state_collect_20261012.py:20-44,49-57`), while the child guard binds the exact replay content and all versioned project files (`scripts/child_input_guard_20261012.py:28-52`).

Reset and alignment evidence is consistent: the three `initial_states.npz` files are byte-identical, seven bootstrap-to-first-action state tensors are elementwise identical, the recorded rigid-cache differences are zero, and the applied reference is equal on all common valid frames. The code explicitly does not establish identity of hidden PhysX state (`analysis.json:196-198`; `docs/controller_state_diagnosis_v2_20261012.md:23`).

The first preflight remains a real FAIL: validation exited 1 on missing `upstream_commit` before simulation (`results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json:2-29`). Its hash remains bound in V2 (`versions.json:79-80`). The first V2 artifact verifier also remains FAIL because it incorrectly assumed cases were axis 0 (`.aris/traces/.../artifact_verifier_attempt1_DOCUMENT_WITNESS.json:2-12,84`); the corrected witness records the axis fix without rerunning the documented command (`DOCUMENT_WITNESS.json:23-28`).

The V2 document check has one invocation, exit 0, and the expected sentinel (`DOCUMENT_WITNESS.json:2-21`). The final run has a recorded exit 0 and `CONTROLLER_STATE_DIAGNOSIS_COMPLETE`, but the record explicitly says it is an external tool-session observation and not an automatic server parent receipt (`EXECUTOR_TERMINAL_OBSERVATION.json:2-10`). Complete child returns, logs, outputs, and independent raw recomputation make phantom results unlikely; the missing automatic parent receipt remains a provenance limitation.

The three jobs used 158.556775 seconds against the 600-second round cap, and each recorded child exit code is 0 (`budget_ledger.json:3-21`). The frozen resource snapshot contains those three rows and reports `CAPS_PASS` (`snapshot_1791782554748181514.json:1966-1989,2075-2120`).

## D. Dead Code Detection: PASS

The metric is live. The base collector calls `metrics` (`scripts/physics_guided_collect.py:284`); V24 substitutes the V23 metric import (`scripts/contact_com_collect_v24_20261012.py:9-12`); V28 removes only dense-contact recording and retains the scoring path (`scripts/contact_com_collect_v28_20261012.py:4-16`). `metrics` calls `surface_metrics` and computes all exported gates (`scripts/physics_guided_metric_v23_20261012.py:103-160`). Analysis calls `metrics` for all 72 rows per state and then `compare_metrics` (`scripts/controller_state_diagnosis_20261012.py:117-142`).

`bounded_update`, `rank_candidate`, and `unit_witness` do not affect this controller-state pipeline. `endpoint_time_error_s` is live in `condition_pass`, but it is absent from the six-number state mean table. The criterion breakdown exposes its failure count, which should remain visible in any cross-state description.

## E. Scope Assessment: WARN

The experiment covers 24 internal validation sources, two families (12 jump and 12 dance), three horizontal offsets, one seed, and three checkpoint states. The manifest contains 19 upstream-train and 5 upstream-test sources. It is not an independent final test. All three checkpoints belong to the same CLoSD algorithm (`results/controller_state_diagnosis_v2_20261012/versions.json:56-79`; `preparation.json:101-113`; independent trace `:66-77,132-153`).

The design supports a narrow controller-state sensitivity diagnosis. It does not compare three independent algorithms and cannot support claims of comprehensive coverage, robustness, generalization, algorithmic superiority, innovation efficacy, downstream learning, or robot utility. The source and docs correctly encode this boundary (`scripts/controller_state_diagnosis_20261012.py:145-149`; `docs/controller_state_diagnosis_20261012.md:3-5,25-29`; V2 doc `:3-5,21-23`).

## F. Evaluation Type: WARN

Primary classification: **simulation_only**. Secondary classification: **synthetic_proxy tracking**, with dataset-derived endpoint and heading targets plus simulated contact/surface/continuity proxies. There is no human evaluation, real-robot execution, independent semantic evaluator, or real physical-contact ground truth. `versions.json:56-60`, `physics_guided_collect.py:135-138`, and `physics_guided_metric_v23_20261012.py:100,160` state these limits.

The evaluation labels are honest enough to avoid a FAIL. They require continued qualification anywhere the numbers appear.

## Endpoint-time semantic audit: WARN

`endpoint_time_error_s` does the following (`scripts/physics_guided_metric_v23_20261012.py:124-133`):

1. Compute distance to the endpoint target during the generated segment.
2. Find that trajectory's own minimum distance.
3. Keep frames within 1 mm of the minimum.
4. Choose the retained frame closest to frame `expected-1`.
5. Report the time difference from `expected-1`.

This is a near-minimum-distance timing proxy. It is not first entry into the 0.1 m target region, it is not deadline position success, and it should not be described as arrival lateness. The existing report builder already states this correctly (`scripts/build_contact_measurement_report_20261011.py:468-470`).

| State | Ever enters 0.1 m | Final position pass | Time-proxy pass | Final-position pass but time-proxy fail | Position+heading pass but time-proxy fail | Time-proxy pass but final-position fail |
|---|---:|---:|---:|---:|---:|---:|
| multitask | 27 | 15 | 32 | 6 | 3 | 23 |
| t2m | 31 | 15 | 35 | 6 | 3 | 26 |
| unfinetuned | 33 | 16 | 33 | 6 | 6 | 23 |

These differences show that the field cannot carry ordinary arrival semantics. Final endpoint position and heading are separately required (`metric_v23:148-160`), so a time-proxy pass alone cannot create a condition or joint pass. The proxy can reject a row that satisfies final position and heading, so it materially affects the condition counts. For a trajectory that ends before the generated segment, the fallback uses `expected*dt`, whereas the normal anchor is `expected-1`; joint success still requires completion, but component summaries should preserve this detail.

## Remaining source-level limitations

The current bytes pass all independent checks, including the new publication guard. The following source-level limitations remain and should not be erased by that successful current check:

- Preparation copies the parent baseline and freezes the new copy's hash, but the original preparation source does not compare the parent file to the parent's recorded baseline hash (`controller_state_diagnosis_20261012.py:35-56`). This audit independently verified the current chain.
- `initial_states.npz` is checked and retrospectively hashed by analysis, but it is absent from the collector lineage output map. The supplemental publication guard now verifies the metadata/analysis binding (`controller_state_publication_guard_20261012.py:50-75`), which hardens publication of the current result but does not change the original execution-time lineage.
- The upstream cleanliness command checks tracked changes under `closd`; it does not detect untracked files or changes elsewhere in the upstream checkout (`controller_state_collect_20261012.py:24-27,51-52`).
- Many integrity gates use Python `assert`. The recorded launcher did not use optimized Python, but explicit exceptions would be stronger for future fail-closed execution.
- Exported reset tensors and caches match; hidden PhysX state remains outside the certified scope.

## Claim impact

- **Supported with scope:** the exact counts and means occurred under this frozen simulation replay.
- **Supported as descriptive only:** the frozen criterion breakdown locates overlapping observed barriers without changing the rules.
- **Needs qualification:** the three checkpoint states differ numerically, showing sensitivity inside one algorithm and this one-seed internal set.
- **Unsupported as worded:** `endpoint_time_error_s` measures arrival time, deadline success, or lateness.
- **Unsupported:** changing controller training state resolves the physical-quality lower bound; every state has zero joint passes.
- **Unsupported:** innovation efficacy, independent algorithm superiority, robustness, generalization, semantic validity, downstream value, or robot utility.

## Action items

1. Keep `simulation_only` and `synthetic_proxy tracking` next to every presentation of these results.
2. Call `endpoint_time_error_s` a near-minimum-distance timing proxy. If arrival semantics are needed later, freeze a separate first-entry or deadline-position event without rewriting these historical scores.
3. Include `endpoint_time_error_s` and `support_observations` in future cross-state summaries.
4. Limit the conclusion to the one-seed, 24-source, internal mixed-split diagnosis of three states within one algorithm.
5. For future runs, create an automatic server-side parent completion receipt and terminal traces for post hoc diagnostic scripts.
6. Bind the parent reference provenance, initial-state archive, and relevant upstream files at execution time using explicit exceptions rather than assert-only checks.

## Audit trace and hashes

The full trace is `.aris/traces/controller_state_diagnosis_v2_20261012/integrity_audit/`. It contains the independent implementation, successful JSON/stdout/exit records, preserved failed audit attempts, and the independent publication-guard recheck. The complete audited hash map is in `independent_recompute.json:3-45`; selected hashes are duplicated in `EXPERIMENT_AUDIT.json`.

Raw NPZ archives remain server-only under `/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/controller_state_diagnosis_v2_20261012`. No simulator, GPU, generation, training, preparation, experiment check, or experiment run was invoked by this audit.
