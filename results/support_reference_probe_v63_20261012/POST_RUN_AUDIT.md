# Post-run Experiment Audit — V63/V65 Support-Reference Diagnostic

**Date:** 2026-10-12  
**Auditor:** GPT-5.6-Sol, ultra reasoning, fresh same-family reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Audit scope:** completed V65 continuation and one locked, read-only V63 analysis; no experiment, GPU job, score, source, or Overleaf mutation by this reviewer  
**Historical pre-run verdict:** **FAIL retained**  
**Current post-run evidence verdict:** **WARN**

The authoritative remote outputs support a bounded negative diagnosis: under the frozen one-seed, three-arm protocol, neither repair arm increases physical-pass or joint-pass count above baseline, and the locked analyzer’s decision to stop this root-translation parameterization is reproduced exactly. This is not an efficacy, equivalence, novelty, acceptance, C2/C3, G1, downstream, or real-world claim.

The evidence is not clean enough for an unqualified PASS. The original V63 document witness remains FAIL; V65 continued under a separate numerical-contract addendum and independent preflight. The server-created continuation receipt remains launch-only, while the exit-0/sentinel record is a later observer transcription. The frozen analysis phrase “no test data” requires the additive clarification that no **internal** held-out test partition was evaluated even though 5 upstream official-test sources are present. The study also remains a one-seed simulation proxy using untracked, hash-frozen sources under divergent Git HEADs.

The preserved pre-run audit was not modified. Its local and remote hashes remain:

- `EXPERIMENT_AUDIT.json`: `cfd74d960036300c2fa3c86178b9027364abf13e9c2a9ee1d8c6d2c13400420f`
- `EXPERIMENT_AUDIT.md`: `ac65008a0b96b3d3f4881fc9af10ed55d16ef404e7c611c46f592b16a509f5ab`

## A–F verdicts

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | **PASS** | Dataset-derived target/heading are separated from a prior model/simulator-derived reference. Tracking and contact are explicitly labeled synthetic/simulation proxies; semantics are unevaluated. |
| B. Score normalization | **PASS** | No reported metric uses a prediction max/min/mean denominator. Raw SI-unit means and finite/missing/total denominators independently match. |
| C. Result existence and lineage | **WARN** | All three authoritative arms, requests, raw outputs, ledgers, summary, and analysis exist and cross-bind. The original FAIL remains. The server receipt is `LAUNCHED`; the terminal exit record is a posthoc observer transcription. |
| D. Dead code and execution | **PASS** | Translation, replay, shared metrics, reset checks, clustered analysis, and decision paths are represented in actual files and independently recompute. |
| E. Scope and fairness | **WARN** | Fair paired scope is 24 sources × 3 offsets × 3 arms, seed 11. The frozen “no test data” wording is ambiguous because the internal validation set contains upstream official-test material; one seed cannot support generalization. |
| F. Evaluation classification | **PASS** | Overall classification is **simulation_only + synthetic_proxy**. Dataset target specification is not full-trajectory real GT; no human or official semantic evaluation occurred. |

## A. Provenance

Preparation reads V51 `raw/val_shared_s11_r0`, constructs the baseline reference from that run’s simulated `actual` trajectory, and derives both repairs deterministically from simulated velocity, body-force, and collision-geometry evidence (`D:/具身智能/conditional-human-motion-generation/scripts/prepare_support_reference_probe_v63_20261012.py:19-59`). The protocol expressly says the tracking target is a simulated refined synthetic reference, not MoCap truth (`D:/具身智能/conditional-human-motion-generation/docs/support_reference_probe_v63_20261012.md:3-9`).

The endpoint target and heading are dataset-motion-derived and fixed before generation. The raw metadata states `goal_provenance = "dataset motion, fixed before generation, endpoint at 4 seconds"`; all arms carry the same `original_goal_sha256 = d008fae01d78f5f8f752aa4791b824540aec5791fd36173306881bcb09584809`. The metric uses this fixed target at `physics_guided_metric_v23_20261012.py:103-131` and never substitutes the repaired reference as the endpoint goal.

No independent semantic evaluator was run. Every raw row says `semantic_status = "not_verified_by_independent_evaluator"`. There is no official physics/contact evaluator for this custom replay diagnostic; it must not be presented as official HumanML3D benchmark evaluation.

## B. Metrics and normalization

The contact routine transforms body-local collision primitives and the recorded body-local COM into world coordinates, forms the lever from COM to the lowest primitive surface point, and evaluates `v_COM + omega × lever` (`D:/具身智能/conditional-human-motion-generation/scripts/physics_guided_metric_v23_20261012.py:36-100`). It conditions its slip fraction on a fixed force/proximity support mask. This is a normal rate denominator, not normalization by a prediction maximum, minimum, or mean. The output is explicitly named `surface_slip_proxy_fraction`.

The analyzer reports raw means and, for each of seven metrics, `finite`, `missing_or_nonfinite`, and `total` (`D:/具身智能/conditional-human-motion-generation/scripts/analyze_support_reference_probe_v63_20261012.py:42-48`). Independent calculation from all 216 raw rows reproduced every float exactly. Each arm has 72 finite and 0 missing/nonfinite values for each reported metric.

| Arm | endpoint error m | heading error rad | endpoint-time error s | slip proxy fraction | penetration fraction | jerk p95 m/s³ | tracking mean m |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1959069926 | 0.2121799644 | 0.5833333638 | 0.5424111457 | 0 | 1611.0568736 | 0.0550110892 |
| unconstrained | 0.1940759914 | 0.2071223757 | 0.5606481774 | 0.5544018713 | 0 | 1607.7525681 | 0.0567622737 |
| endpoint preserved | 0.1949156253 | 0.2039686050 | 0.5370370650 | 0.5538484309 | 0 | 1599.0761909 | 0.0568203655 |

These are descriptive proxy means. No efficacy inference is made from their direction.

## C. Terminal artifacts, lineage, and counts

The authoritative remote directory contains exactly three requests and three raw arm directories. Each arm has 72 unique cases, 24 unique sources, and the three offsets `[-0.15, 0, 0.15]` for every source. All 216 cases completed, all 25,920 transition-valid flags are true (8,640 per arm), all reference-frame arrays equal 0–119, and no case terminated.

| Arm | completed | condition pass | physical pass | tracking pass | joint pass |
|---|---:|---:|---:|---:|---:|
| baseline | 72 | 5 | 0 | 68 | 0 |
| unconstrained | 72 | 6 | 0 | 67 | 0 |
| endpoint preserved | 72 | 4 | 0 | 67 | 0 |

Every count matches `summary.json:$.reports.*.counts`; every raw output hash matches its execution-bound `lineage.json`; every request equals the lineage request; every receipt binds its trajectory hash. The analysis input hashes match all three request, metadata, metrics, lineage, trajectory, observation-trace, and initial-state files. Key hashes are:

- analysis: `b1ed5679b672e7375c8c79ca0e3a76a97943dc142e330ea21431f95505659a7e`
- summary: `a504f56cf37dec79fb46aeace51a80a25fc1dbbb8f4251bd8727ffeb881193b2`
- baseline / unconstrained / endpoint metrics: `fb944ae…`, `9e318a07…`, `dcdc1ba2…`
- baseline / unconstrained / endpoint trajectories: `7802cdb9…`, `67876934…`, `c7371a9b…`
- shared initial-state archive: `b25404dd3cfb63710078bf7dbb824e3749b1cdebabd68e7719c2f6b3b12617fe`

Baseline was not rerun. Its receipt predates V65 preflight, its hashes are unchanged from the original and continuation witnesses, and the main ledger contains one baseline row followed by exactly two comparison rows. V65 reuses `runtime_witness.json` for baseline at `continue_support_reference_probe_v65_20261012.py:80-89`.

Terminal evidence has one residual weakness. `state.json:$.status` and `summary.json:$.status` are `COMPLETE_REFERENCE_SIDE_DIAGNOSIS`; both comparison receipts are complete; PID 2891330 is absent; and `PARENT_EXIT_OBSERVATION.json` records tool session 34005, chunk `bd448f`, exit 0, and `SUPPORT_REFERENCE_V65_CONTINUATION_COMPLETE`. But `continuation_v65_receipt.json:$.status` remains `LAUNCHED`, and the observer file states that it is not a server-created durable parent receipt. This audit therefore retains C = WARN and does not relabel the launch receipt.

The analysis artifact is not a self-contained terminal or budget certificate. Its input map hashes requests, selected raw artifacts, summary, versions, and the plan lock, but omits arm receipts, the continuation receipt, witnesses, and both ledgers (`analyze_support_reference_probe_v63_20261012.py:49-51`). This audit checked those artifacts separately.

The public local mirror is intentionally metadata-only. The local result tree has 53 files and the remote tree has 74; all 51 common repo-relative files have identical hashes. The local-only files are `baseline_metadata_for_witness.json` and the root-level public archive. The remote-only set comprises 22 deliberately excluded authoritative files plus the same archive under `public_exports/`. The export manifest declares 47 payload files, all 47 verify inside its 48-member archive (payload plus manifest), and all 43 V63-result payload paths exist locally with matching hashes. Licensed motion/reference/trace arrays are explicitly excluded and were audited remotely. The local and remote archives share SHA-256 `d0e6be9753e7f8a75e2a334bd7376e795ec57c3dfaa31348b346640ea70be59d`. This is a documented public-mirror limitation, not literal full-tree parity.

## D. Executed code paths

`translation()` is defined and fixture-tested at `support_reference_repair_core_v63_20261012.py:10-54`; preparation invokes it for both repair arms at `prepare_support_reference_probe_v63_20261012.py:30-59`. The prepared arrays show nonzero bounded shifts, so the function is not dead: maximum horizontal shift is 0.0607537851 m for unconstrained and 0.0607520379 m for endpoint-preserved, below the fixed 0.10 m bound.

V65 invokes `probe.batch()` for the two unused arms only after rechecking the frozen evidence before each arm (`continue_support_reference_probe_v65_20261012.py:68-91`). The shared metric routine emits the raw fields found in every `metrics.json` (`physics_guided_metric_v23_20261012.py:103-160`). The analyzer copies pass counts from `summary.json` and uses those aggregates for the physical-improvement branch (`analyze_support_reference_probe_v63_20261012.py:45-48,66-71`) instead of recomputing them from raw rows. Its source-cluster loop also reuses baseline source indices without explicitly asserting cross-arm `source_id` equality (`:52-59`). Independent count and identity recomputation matches exactly here, so the STOP decision is numerically correct for the frozen files, but future analysis should derive decision counts from primary rows and assert all pairing keys.

The analyzer is bound before outcomes by `analysis_plan_lock.json` SHA-256 `0452de2471d4804226bf7913c3de356483d9784aa16db2fac5afc87beb6c8b27`; the output's `source_sha256` equals the current analyzer SHA-256 `2a76407d53b7f8fe7dfe823eebb327a5b98b7da746d40dec73e4905150ddc11d`. The lock binds analyzer code identity only. It is not a complete preregistration of expected input hashes or metric specification, although file chronology is favorable: lock 03:02:44 UTC, V65 launch 03:07:05, and comparison receipts 03:07:53/03:08:43. Integrity-critical gates throughout preparation, runtime, analysis, and continuation use `assert`; the documented commands did not request optimized Python, so no bypass is observed, but `python -O` would remove those checks.

## Reset, freeze, and reference invariants

All requests are identical in config, manifest, split, mode, seed, round, noise round, thresholds, and versions; only the replay-motion hash differs. Metadata match across arms for case/body order, goals/headings, foot geometry, native body properties, coordinate system, simulator parameters, all listed seeds, policy observation mode, controller state, and velocity rule.

All three `initial_states.npz` files have the exact same SHA-256. For each arm, the seven exported bootstrap tensors equal their first-action tensors bitwise. All exported arrays, including first-action contact force, are bitwise identical across arms. Every `initial_reset_gate_pass` is true and every named rigid-cache discrepancy is zero. The current analyzer does not enforce the expected four cache keys and would accept an empty cache-consistency map because `all([])` is true (`analyze_support_reference_probe_v63_20261012.py:32-35`); actual metadata contain the four expected zero-valued entries, so this is a future schema-hardening issue. These checks cover exported tensors and declared caches only; they do not establish hidden PhysX state equality. The analyzer states that limitation at lines 28–40 and in `analysis_*.json:$.reset_checks`.

All frozen references have shape `120×72×24×3` and carry the exact same goals. Frames 0–60 and Z are exact across variants. Relative-body roundoff is at most `1.1920928955078125e-7 m`. Endpoint-preserved final pose is exact; the unconstrained final translation changes by design. All original targets remain unchanged.

Time alignment is also limited. Reference/observation at transition index `k` are recorded before action `k`; actual state `k` is recorded after the action. This is same-transition-index evidence, not simultaneous pre-action sensor truth (`D:/具身智能/conditional-human-motion-generation/docs/support_reference_continuation_v65_20261012.md:19`). The retained V63 document's broader same-time wording at line 25 must not be reused; V65's transition-index statement is the authoritative limitation.

## Contact point, frame, and COM assumptions

No new frame-mapping defect was found in the implemented proxy. Body-local primitive points and the local COM are rotated into world coordinates; the lowest collision-surface point is used; and the COM-corrected material-point velocity formula is consistent with the frozen convention that `linvel` is COM velocity.

The measurement still has strict limits. The force is a body-level simulator force, so it does not locate the physical contact point. The lowest primitive point may differ from the force-application point. The COM-velocity convention relies on the prior frozen calibration and was not rerun by this audit. The metric therefore remains a collision-geometry/body-force surface proxy, not contact-manifold truth.

## E. Paired analysis and scope

The analyzer clusters by original source, not by the 72 correlated offset rows (`analyze_support_reference_probe_v63_20261012.py:52-64`). Independent recomputation with 24 source clusters, three offsets per source, `RandomState(6211)`, and 5,000 resamples reproduces both arms exactly:

- joint gain: 0.0 percentage points
- 95% source-cluster interval: `[0.0, 0.0]`
- all 24 per-source changes for each arm: 0

Because physical-pass count is also 0 for all arms, the locked rule reproduces `STOP_THIS_ROOT_TRANSLATION_PARAMETERIZATION_NO_PHYSICAL_OR_JOINT_GAIN` (`analyze_support_reference_probe_v63_20261012.py:65-78`). This is a sample-bound mechanism decision. The all-zero interval does not prove population equivalence.

The frozen phrase “no test data” in the analysis scope is accurate only for the study’s internal split: all 72 cases are internal `val`. The upstream official split is 19 train sources/57 inputs plus 5 test sources/15 inputs. `SCOPE_CLARIFICATION.json/.md` states this accurately and says pretrained exposure is unverified, while preserving the original analysis. This clarification is additive; it cannot make the frozen wording retroactively precise. E remains WARN.

## F. Evaluation classification

- Endpoint, heading, and timing: simulation-only execution against a dataset-derived target specification; this is not full-trajectory real GT.
- Tracking: synthetic proxy against a prior simulated/model-derived trajectory.
- Contact, penetration, and jerk: simulation-only; slip is explicitly a geometry/body-force proxy.
- Semantics: not independently evaluated.
- Human evaluation: none.

The overall classification is **simulation_only + synthetic_proxy**.

## Budget and timing scope

The main ledger has exactly three successful rows:

- baseline measurement: 50.4387053065 s
- unconstrained comparison: 50.1901863068 s
- endpoint-preserved comparison: 50.1503653433 s

Main wall time is 150.7792569567 s. Raw simulator receipts sum to 142.0064954758 s. The separate numerical-contract ledger records 0.9037324842 s of saved-array CUDA arithmetic with zero simulator/generator calls. Combined charged time is 151.6829894409 s, below the 600 s own cap. V65 includes the separate numerical charge in its own-cap reservation (`continue_support_reference_probe_v65_20261012.py:27-33`), but the inherited global and per-stage arithmetic reads ordinary `budget_ledger.json` rows only (`run_support_reference_probe_v63_20261012.py:35-38`). Omitting 0.9037324842 s cannot approach the 72-hour or stage caps in this run, but the gate implementation is incomplete.

The analysis JSON’s `simulation_calls = 0` and `generation_calls = 0` describe the read-only analyzer invocation. They do not replace the study total in `summary.json`, which records three simulator batches. Any report must present both scopes together.

## Historical witness and additive remediation

The original `DOCUMENT_WITNESS.json` remains `status: FAIL` for `applied_reference_canonical_exact` and `reference_velocity_successive_exact`; `summary.json:$.original_document_verdict` correctly says `FAIL_RETAINED`. The separate numerical addendum passes the pre-existing `1e-12 m` root tolerance and exactly reproduces the executed float32 CUDA velocity expression. It does not convert the original witness to PASS and does not support efficacy. Its 0.9037324842 s CUDA arithmetic charge remains separately disclosed.

## Reproducibility

Local Git HEAD is `ffadb70f3f4cb7de2537529ba26a66bc6d55ee66`; remote HEAD is `e1ef719d6702d15a5ab211db0e55be438b2d1c88`. The audited V63/V65 files are untracked on both checkouts. Direct source hashes match across hosts, including analyzer `2a76407d…` and V65 continuation `a801b769…`, so this audit can bind exact bytes. Neither Git HEAD alone reproduces those bytes.

## Claim impact

- **Supported:** three-arm completion, 72 cases per arm, no baseline rerun, exact raw counts/means/denominators, execution-bound output hashes, paired 24-source calculation, exported reset equality, and the frozen sample-bound STOP decision.
- **Supported with explicit limits:** consistent contact-point/COM arithmetic inside its proxy definition; same-transition-index timing; posthoc observer evidence for parent exit 0.
- **Needs qualification:** “no test data” means no internal held-out test partition, while upstream official-test material is present.
- **Unsupported:** physical or joint improvement, population equivalence, untouched official-test generalization, unseen-source generalization, real contact truth, semantic validity, novelty, C2/C3 validity, G1 readiness, downstream benefit, or acceptance.
- **False if asserted:** that the original V63 document witness passed.

## Required actions

1. Keep the pre-run audit and original document-witness FAIL immutable; cite this post-run file as an additive audit only.
2. Preserve the terminal state, summary, per-arm receipts, lineages, and hashes. Label `PARENT_EXIT_OBSERVATION.json` as posthoc observer evidence and leave `continuation_v65_receipt.json` as launch-only.
3. Replace or qualify “no test data” with “no internal held-out test partition evaluated,” disclose the 19/5 upstream train/test source mix, and state that pretrained exposure is unverified.
4. Pair analyzer-local `simulation_calls = 0` with the experiment’s three simulator batches and the separate numerical CUDA charge.
5. Keep tracking/contact labeled synthetic/simulation proxies and keep transition-index timing distinct from simultaneous sensor truth.
6. Do not infer efficacy, equivalence, novelty, C2/C3 validity, G1 readiness, downstream benefit, or real-world physical validity from this one-seed diagnosis.
7. For future analyses, hash arm receipts, the continuation receipt, witnesses, and both ledgers; recompute decision counts from raw rows; assert cross-arm source/clip/offset identity; require the expected reset-cache keys; include the separate numerical ledger in global/stage gate arithmetic; and replace integrity-critical `assert` statements with explicit fail-closed checks.
8. Archive the untracked hash-frozen sources with both Git HEADs before treating a repository revision as full reproduction metadata.

## Primary artifact hashes

- `analysis_1791774541453036805.json`: `b1ed5679b672e7375c8c79ca0e3a76a97943dc142e330ea21431f95505659a7e`
- `summary.json`: `a504f56cf37dec79fb46aeace51a80a25fc1dbbb8f4251bd8727ffeb881193b2`
- `budget_ledger.json`: `c1a19bd9200435ff926cf60ef577d7b570946a36b52ee7a3688395d2beeb48c0`
- `continuation_v65_receipt.json`: `b76e4b3a2081b7a4f96acac14517ebbda73b37e96888d3449fa3f106f4bf1b5b`
- `PARENT_EXIT_OBSERVATION.json`: `b9ba36d4c1f96534e08262ef32b54363872e57c88e82b336e6f9bbb8cb15e3f3`
- `SCOPE_CLARIFICATION.json/.md`: `b19376e6…` / `b59ac5b6…`
- `analysis_plan_lock.json`: `0452de2471d4804226bf7913c3de356483d9784aa16db2fac5afc87beb6c8b27`
- analyzer / V65 source: `2a76407d…` / `a801b769…`
- original witness / numerical addendum: `c8f32067…` / `fc990207…`
