# Experiment Integrity Audit

**Date:** 2026-10-12  
**Project:** paired-feedback validation V51 with append-only V53–V55 analysis  
**Auditor:** GPT-5.6-Sol ultra, fresh same-family reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Overall verdict:** **FAIL**  
**Integrity status:** **fail**

The frozen V51 evaluation fails the mandatory provenance rule. Its tracking evaluator uses a model/simulator-derived, physics-refined trajectory as `reference`, then contributes that metric to `tracking_pass` and `joint_pass`, without explicitly labeling the reference as a synthetic proxy. The later provenance document and V53/V54 outputs disclose this correctly, but append-only clarification cannot retroactively repair the original record.

V55 separately reconstructs the current selected motions and `construction_complete` labels from execution-bound raw candidate artifacts and replay-bound motion. This supports the current 214/213 construction-complete totals and 0/0 end-to-end totals with qualifications. It does not change the historical V51 failure, establish efficacy, or establish acceptance.

## A–F verdicts

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | **FAIL** | V51 uses a generated/simulator-refined trajectory as tracking `reference` without an explicit `synthetic_proxy` label. |
| B. Score normalization | **PASS** | Fixed denominators are 216 per method and 72 per seed. No prediction extrema or mean supplies the denominator; raw counts and coverage are present. |
| C. Result existence and counts | **PASS** | Terminal summary, 21 successful ledgers, 21 requests, 21 raw batches, six selected arm/seed groups, V53/V54 outputs, V54 receipt/log, and V55 certificate exist and match the authoritative remote files. |
| D. Dead code | **WARN** | The V51 runner imports `compare` but does not call it; the V23-local ranker is not used by this path. Live metrics do appear in raw outputs. |
| E. Scope | **WARN** | Actual scope is explicit and limited, but `versions.json` retains contradictory V28 test/44/30 fields. The custom `val` set is not the official HumanML3D validation split. V53 also drifted from the configured bootstrap count; V54 corrected it. |
| F. Evaluation type | **PASS for current disclosure** | Composite evaluation is `simulation_only`; tracking is `synthetic_proxy`; feedback is a `self_supervised_proxy`; only the condition target is dataset-derived in a narrow sense. No human or real-robot evaluation ran. |

## A. Provenance failure

The original frozen path is unambiguous:

- `scripts/physics_guided_queue.py:144-155` selects a candidate and, unless `raw=True`, copies `arrays['actual']` into a selected NPZ key named `reference`. The only source label is `physics_refined`.
- `scripts/run_paired_feedback_validation_v51_20261012.py:161-163` invokes this default refined path.
- `scripts/physics_guided_metric_v23_20261012.py:118,136-160` loads that reference, computes tracking errors, and feeds them into `tracking_pass` and `joint_pass` without a proxy qualifier.
- The same metric correctly identifies surface/contact evidence as a proxy at `scripts/physics_guided_metric_v23_20261012.py:93-100`, showing that the missing tracking qualifier is specific rather than a general naming convention.

The appended `docs/paired_feedback_v51_evaluation_provenance_20261012.md:3,7-15` accurately records that it was added after the audit finding, classifies tracking as synthetic proxy and physical evidence as simulation-only, and limits the result to 24 custom validation sources. Its SHA-256 is `a160b2b8c0644996f08e9a483903e61f59e529b87b0255d436f28ab8cb5bc4ba`.

An official text-motion evaluator implementation and weights are listed in `results/paired_feedback_validation_v51_20261012/versions.json:31,44,51-56`, but no official semantic/diversity evaluation ran. V53/V54 correctly leave that guard incomplete.

## B. Normalization

No model-self-normalization was found.

- `scripts/analyze_paired_feedback_v51_20261012.py:72-89` uses raw binary outcomes, exactly 216 observations per method, nine observations per source, and `/72` for seed-specific percentage-point differences.
- Lines `106-111` emit raw means together with `recorded_finite` and `missing_or_nonfinite`; every reported field has 216/216 finite coverage per arm.
- `scripts/physics_guided_metric_v23_20261012.py:164-171` normalizes candidate-ranking errors by frozen thresholds, not by statistics of the predictions being scored.

The event-localized `endpoint_time_error_s` at metric lines `124-131` is an interpretation caveat, but it is not a prediction-derived score denominator.

## C. Terminal evidence and counts

The authoritative remote V51 tree is terminal and internally complete:

- PID `2851799` was absent on final inspection.
- `state.json` is `COMPLETE_VALIDATION_QUALITY_FLOOR`.
- `summary.json` exists.
- `budget_ledger.json` contains exactly 21 unique entries, all exit code 0: 20 comparison entries plus one measurement entry, totaling `1049.6597477272153` seconds.
- Requests: 21 JSON files.
- Raw output: 21 directories × 10 files = 210 files.
- Selected output: six arm/seed groups × JSON, lineage, and NPZ = 18 files.
- Construction/replay batches: 15/6.
- The V54 invocation log now exists locally and remotely with SHA-256 `575a90ecdc6c7bd57aa292c6d611a3f3802c3c8d6dfc607d68735d3606e3b28d`.

Every request hash, execution-bound lineage, receipt, metadata/metrics/trajectory hash, case count, and replay count matched the authoritative remote files. Raw arrays across all 21 batches have consistent shapes: position/reference/linear velocity/angular velocity/contact force `(120,72,24,3)`, quaternion `(120,72,24,4)`, valid mask `(120,72)`, terminated `(72)`, and reference frame `(120)`.

No inspected trajectory array contains a reward or legacy-cache key. Sidecar hashes for initial state, observation trace, sensor diagnostic, and native contacts match metadata; every `native_contacts.json` has empty records/errors/dtype fields. This establishes absence in the inspected outputs, not a claim about uninspected runtime memory.

Key terminal hashes are recorded in `AUDITED_INPUT_HASHES.json`; the full 276-file audited input set has canonical digest `395b8ca4a2cb12e1af6ab67ec5edae711e6676c97eb32294d6854e89608c4d04`.

## Pairing, budget, and completion guards

The requested protocol mechanics are supported:

- Both arms share ordinal 0 at `run_paired_feedback_validation_v51_20261012.py:152-155`.
- Ordinals 1–2 use matching candidate-ordinal noise, with equality asserted at lines `155-160`.
- Original goals, headings, source IDs, families, offsets, and generation-noise hashes match across arms.
- Each method receives three candidates per input.
- The realized design is 24 sources × three offsets × three seeds = 216 inputs per method.
- Reset/native-velocity handling is present: `reference_velocity_collect_v43_20261012.py:13-36` uses zero first-frame velocity, successive applied canonical positions divided by `s_dt`, and repeated-frame handling. The reset ancestor performs the official reset and verifies exported reset/cache equality.
- Raw versus refined selection is explicit at `physics_guided_queue.py:145-155`; V51 uses the refined path.

Main V51 totals are replay totals (`run_paired_feedback_validation_v51_20261012.py:169-178`). V53/V54 separately define `end_to_end_joint_pass = construction_complete AND replay joint_pass` at analyzer lines `56-60`.

| Arm | Construction complete | Replay completed | Replay joint pass | End-to-end joint pass |
|---|---:|---:|---:|---:|
| Feedback | 214 | 215 | 0 | 0 |
| Random | 213 | 214 | 0 | 0 |

## D. Dead paths

- `run_paired_feedback_validation_v51_20261012.py:126` imports `compare`, but `main()` never calls it.
- `physics_guided_metric_v23_20261012.py:164-171` defines `rank_candidate`, while the V51 queue uses the implementation imported from `physics_guided_core.py` at `physics_guided_queue.py:12,144`.

Tracking, condition, physical, completion, and joint fields are live: they are created at metric lines `118-160` and appear in every relevant metrics output.

## E. Scope

The actual scope is one custom validation configuration: two families, 12 jump and 12 dance sources, offsets `-0.15/0/+0.15`, seeds `11/12/13`, nine rows per source per arm, 216 inputs per arm, 15 construction batches, and six independent replay batches. There is no independent test, G1 run, downstream training, scene sweep, or multi-configuration evaluation.

The custom `split="val"` rows contain 19 HumanML3D official-train rows and five official-test rows, with zero official-val rows. “Validation” therefore names this project split, not the official HumanML3D validation split.

`versions.json` retains contradictory inherited V28 fields: 44 construction/30 replay at lines `66-68`, 432 test inputs/48 test sources at lines `78,84`, and “complete paired test” at line `87`, alongside the correct V51 validation-only 24-source and 15/6 fields at lines `99-106`. The V51 protocol document is accurate at `docs/paired_feedback_validation_v51_20261012.md:3-7,25`.

V53 used 10,000 bootstrap samples at analyzer lines `84,114`, while the frozen configuration specifies 5,000 at `configs/physics_guided_20261011.json:38`. V54 corrects the current analysis to 5,000.

## F. Evaluation classification

| Component | Classification |
|---|---|
| Condition target | `real_gt` only in the narrow dataset-derived-target sense; it is not robot-execution ground truth or an official benchmark endpoint |
| Tracking | `synthetic_proxy` |
| Feedback/correction | `self_supervised_proxy` using prior simulated execution error |
| Physical metrics | `simulation_only`; contact and surface-slip are proxies |
| Composite joint result | `simulation_only` |
| Human evaluation | absent |
| Real-robot evaluation | absent |

## Analysis lineage by generation

### V53 — historical warning

- Source SHA-256: `8f99e1901ca091003dc431f902c525eaa877877736982447311a677278be8eec`
- Output SHA-256: `30d61febbcacf0c7e123bc42d8e1feefd5e4640e890b61ac67ef337849a540db`
- V53 correctly reports current counts, raw-versus-refined scope, proxy classifications, missing semantic/diversity guard, validation-only status, and no acceptance.
- Its `input_hashes` are current snapshots. `analyze_paired_feedback_v51_20261012.py:44-60` did not compare replay and selection inputs with execution-time bindings.

### V54 — partial historical remediation

- Source SHA-256: `8e3762c5c381117461c0d578550148ba3228c78d327b3c3daa1aa2375383b8dc`
- Analysis SHA-256: `bebec154baf929d372cd30dcfe3d8011a1d57e1b510950004031c28f6a895b80`
- Execution certificate SHA-256: `e0049f5c7be60a7b06eaf5fe63f91c7a7fe416d973d05331e823af2b8890a9ab`
- Invocation log SHA-256: `575a90ecdc6c7bd57aa292c6d611a3f3802c3c8d6dfc607d68735d3606e3b28d`

V54 verifies all 21 execution-bound batches, replay summary records, selected NPZ motion, candidate output hashes, frozen inputs, 15/6/24/72 scope counts, and the configured 5,000 bootstrap count. Historically, V54 alone still trusted mutable selected-lineage JSON for construction labels, so its certificate did not by itself close the construction-label binding gap.

### V55 — current-state PASS with qualifications

- Source SHA-256: `d22c6ce574d458c47f83cb62160ecbca72b66ab28e6a09b7689b8da72897f42d`
- Certificate SHA-256: `7800698a34fdcb58710c81afc0ad17a2814c572b4fd6ffb826081d2939a685b0`

V55 uses the correct trust direction for current-state evidence:

- `verify_selected_construction_v55_20261012.py:32-42` sends raw candidate batches through `guard_batch`.
- Lines `51-53` recompute the selected index using the frozen ranker.
- Lines `53,58-59` obtain `construction_complete` from the selected execution-bound candidate metric.
- Lines `54-57` rebuild the selected reference from the bound candidate `actual` trajectory and valid mask.
- Lines `61-63` compare the rebuilt reference and original goal exactly with the selected NPZ.
- Lines `45-48` require the selected NPZ motion hash to equal the replay request’s execution-bound `replay_motion`.
- The mutable selected JSON is read only for an equality check at lines `43-44,60`; selected-lineage JSON is not trusted.

All 84 V55 certificate input hashes currently match the remote files. All six arm/seed entries report `exact_reference_reconstruction=true` and `selection_labels_reconstructed_from_bound_raw=true`. The reconstructed construction totals are feedback `72/72/70 = 214` and random `72/71/70 = 213`; end-to-end joint is 0 for both arms.

Remaining limits:

1. V55 is a current reconstruction, not proof that the original selected JSON had an immutable execution-time hash.
2. V54 does not bind the V55 certificate; cite the two artifacts together.
3. Most V55 checks use `assert`, but the certificate does not attest that Python optimization was disabled.
4. The certificate omits some request, config, core, and queue dependency hashes.
5. V55 does not reassert a common original-goal hash across all three candidates.
6. Replay rows are combined by positional `zip`, not an explicit `case_id` bijection. This cannot affect the present all-zero replay joint result, but it must be fixed before using a nonzero outcome.
7. Exact family composition, offsets, and custom-versus-official split remain outside V55.

## Git provenance

- Local HEAD: `f0e3c4014428dedcc3ce150e6069af4837149cd8`
- Remote execution checkout HEAD: `e1ef719d6702d15a5ab211db0e55be438b2d1c88`
- The heads differ. Remote analysis files were explicitly copied into the checkout and remain outside its commit identity.

The remote run must therefore be bound by the recorded file hashes, not described as reproduced directly from remote `HEAD`.

## Claim impact

- **Supported:** paired candidate ordinals, unchanged original goals, equal three-candidate budget, 24×3×3 realized scope, 15 construction/6 replay batches, terminal counts, and current-state reconstruction of 214/213 construction-complete totals.
- **Supported descriptively:** feedback records 63 condition passes and random records 39 on this custom validation pilot.
- **Supported prospectively:** V53/V54 correctly call tracking a synthetic proxy and physical evidence simulation-only.
- **Unsupported:** an accepted efficacy improvement or innovation. Both arms have zero replay joint pass and zero end-to-end joint pass; the recorded gain is 0.0 percentage points.
- **Unsupported:** semantic quality, diversity preservation, real physical executability, real contact/slip accuracy, downstream benefit, G1 success, or population-level absence of effect.
- **Unsupported:** calling the original V51 tracking evaluator ground truth or claiming its original proxy labeling was adequate.

The `[0,0]` source-cluster interval is a quality-floor observation caused by all-zero joint outcomes on these 24 custom validation sources. It is not proof of a population-level null effect.

## Required actions

1. Preserve the frozen V51 provenance failure; do not rewrite the original files.
2. Cite V54 and V55 together for current execution/selection evidence. Do not present V54 alone as fully binding construction labels.
3. Before interpreting any future nonzero result, bind the V55 certificate into the analysis manifest, attest optimization mode, save an invocation receipt, emit all dependency/request hashes, and join replay rows by `case_id`.
4. Keep the custom-versus-official split, exact family/offset scope, inherited V28 fields, and remote Git-head mismatch explicit.
5. Keep acceptance false until an appropriately separated test and independent semantic/diversity evaluation exist.

## Audit trace and inputs

- Machine-readable audit: `results/paired_feedback_analysis_v54_20261012/EXPERIMENT_AUDIT.json`
- Audited-input manifest: `results/paired_feedback_analysis_v54_20261012/AUDITED_INPUT_HASHES.json`
- Trace: `.aris/traces/experiment-audit/2026-10-12_run01/`
- This is a same-family provisional audit. It is not cross-family acceptance.
- V56 passive diagnostics are outside this audit’s stated scope.
