# Experiment Audit Report — V28 complete C1

**Date:** 2026-10-12  
**Auditor:** fresh Codex integrity reviewer, read-only, same-family provisional  
**Authoritative server root:** /data/code/ouyanghao/oyh/conditional-human-motion-generation  
**Local public root:** D:/具身智能/conditional-human-motion-generation  
**Hash binding:** 744 audited inputs; canonical manifest sha256:620e5a14127effbd6990ad2127c8a605f8de61baea373bfeaec0494b9ff5aebe

## Overall Verdict: FAIL

The frozen V28 result files themselves correspond and support the negative conclusion: C1 did not improve joint success, every arm has zero joint passes, and G1/downstream work correctly remained unrun. The integrity verdict is FAIL because the reachable resume path accepts a complete receipt without requiring a lineage that was bound at execution, then creates a retrospective lineage. This violates the experiment's explicit “bound complete receipts only” rule. All 74 existing V28 receipts do have execution-bound lineages, so I found no evidence that this weakness altered the current negative result.

A second, non-fatal semantic issue remains: endpoint_time_error_s measures the time of closest geometric approach, with a 1 mm near-minimum tie band biased toward the requested final frame. The frozen prose names a 0.1 s target-time tolerance and a four-second endpoint but does not define this event or a first-entry arrival window. Timing claims therefore require that exact closest-approach definition.

This review is same-family and provisional. The deterministic file, lineage, and numerical checks are independently reproducible; semantic acceptance requires a cross-family reviewer.

## Evidence reviewed

The audit read the frozen code and recursive version dependencies; all requested V28 result tables; all 74 raw batches and their trajectories, metadata, metrics, numeric certificates, receipts, and lineages; every selected, feedback, and request artifact; the full run log; the claim/protocol bundle; the V32/V33/V35 supplemental diagnostics; and the V36 read-only cache certificate. It inspected the bound HumanML3D cache/index paths, model/evaluator weights, and relevant upstream CLoSD state/reset/frame code without copying licensed trajectories.

Independent structural verification covered 10,512 raw metric rows: 44 construction batches plus 30 replay batches. It found 10 arms × 432 replay attempts, 4,320 rows in per_attempt.csv, 48 test sources, three offsets, seeds 11/12/13, all failures retained, 74/74 complete receipts, 74/74 execution-bound lineages, and no hash, case-order, reset, frame, device, contact, COM, replay-motion, or numeric-classification mismatch.

## A. Ground-truth and reference provenance: PASS

The motion input comes from the released HumanML3D cache and is mapped to the simulator from the frozen dataset representation; the target is the dataset motion's final joint location plus the predeclared horizontal offset. Evidence: scripts/physics_guided_collect.py:56-64 and 94-105; results/condition_feedback_formal_v28_20261012/source_manifest.json:1-22. Corrections can change the proposed generation condition within fixed bounds, while scoring retains original_goal and original_heading; evidence: scripts/physics_guided_collect.py:107-139 and scripts/physics_guided_metric_v23_20261012.py:103-123.

No generated motion is presented as physical ground truth. Tracking is against the generated/selected control reference; contact quality is explicitly a simulator/geometry proxy; semantic matching is explicitly a learned proxy. Evidence: scripts/physics_guided_metric_v23_20261012.py:1-5, scripts/physics_guided_semantics.py:1-4, and docs/physics_guided_20261011.md:3-9. Provenance reaches the released cache and official source mapping; it does not independently rederive licensed HumanML3D files from the original raw MoCap archives.

## B. Score normalization: PASS

Persisted performance fields are raw SI quantities. The only endpoint/heading division is candidate ranking by fixed frozen thresholds, not a reported score; evidence: scripts/physics_guided_metric_v23_20261012.py:164-171 and results/condition_feedback_formal_v28_20261012/frozen_thresholds.json:1-20. Final arm means and joint rates use raw fields, while percentage-point gains and source bootstrap intervals are separate outputs; evidence: scripts/physics_guided_analyze.py:39-73. Semantic mean/std assets and evaluator weights are frozen external assets, not statistics of the candidate's own predictions; evidence: scripts/physics_guided_semantics.py:22-35 and results/condition_feedback_formal_v28_20261012/versions.json:34-54.

## C. Result existence and correspondence: PASS

All requested authoritative V28 artifacts exist. summary.json reports 432 attempts and zero joint passes for every arm; proposed is at results/condition_feedback_formal_v28_20261012/summary.json:103-121 and all paired comparisons are zero at :204-295. The numeric gate is false, endpoint-time non-regression fails, and the semantic proxy gate passes at :297-389. progression_gate.json:1-32 independently records zero gain, a [0,0] source-cluster interval, and zero gain for all three seeds. queue_state.json:1-7 is COMPLETE_GATE_NOT_MET with G1 and downstream not run.

The public claim bundle matches these values. docs/candidate_mechanisms_20261011.md:53-59 reports all-zero joint success, the proposed/random endpoint and timing means, the failed gate, and that C2/C3 remain untested. The server and local copies of that document, measurement_review.html, and status.json have identical hashes. No phantom efficacy number was found.

## D. Called metrics and dead code: WARN

The reported V28 metrics are actually called. scripts/physics_guided_run_v28_20261012.sh:14-19 launches the V28 collector; scripts/contact_com_collect_v28_20261012.py:4-16 delegates to V24 while disabling only dense-manifold collection; scripts/contact_com_collect_v24_20261012.py:4-12 rebinds metrics to physics_guided_metric_v23_20261012; scripts/physics_guided_collect.py:284 invokes that rebound function. The formal runner rebinds cached verification to physics_guided_com_lineage_v24_20261012 at scripts/run_condition_feedback_formal_v28_20261012.py:61-74.

Legacy definitions remain unreachable in this pipeline: the V28 scoring path does not call physics_guided_core.heading, foot_geometry, surface_metrics, or metrics; V23 bounded_update, rank_candidate, and unit_witness are also unused; physics_guided_numeric_lineage.py and physics_guided_lineage.certify_cached are not the active certifier. This is dead-code WARN, not evidence that the persisted V23 metrics were fabricated.

## E. Scope, seeds, failures, frozen controls, and budget: PASS with a strict claim ceiling

The documented scope matches execution: 48 preselected test sources, three offsets, three seeds, ten arms, at most three candidates, equal-budget controls, four ablations, and all failures retained. Evidence: docs/condition_feedback_formal_v28_20261012.md:3-7; scripts/physics_guided_queue.py:228-261; results/condition_feedback_formal_v28_20261012/summary.json:3-202. The study is explicitly retrospective/amended and not pristine confirmatory; evidence: results/condition_feedback_formal_v28_20261012/versions.json:55-81.

Analysis clusters the bootstrap by original MoCap source, stratifies by family, uses 5,000 frozen repetitions, checks every equal-budget control and each seed, and separately gates semantic/diversity; evidence: scripts/physics_guided_analyze.py:55-121. Source families and splits are disjoint by original source, but the internal test split contains released-cache examples whose pretraining exposure is not ruled out; docs/physics_guided_20261011.md:3 states that limitation.

The current ledger contains 74 successful task entries plus the retained semantic attempt; results/condition_feedback_formal_v28_20261012/budget_ledger.json:1-525. Together with the prior snapshot at prior_budget_snapshot.json:1-19, the charged total is about 4.01 GPU-hours and approximately 6.99 GB, below the frozen 72-hour and 15 GiB ceilings. These checks support only a frozen single-simulator, single-tracker diagnostic, not robot, downstream, or generalization claims.

## F. Evaluation type classification: PASS

- Condition target provenance: real_gt for the cached dataset endpoint, with a predeclared synthetic ±15 cm intervention.
- Tracking: synthetic_proxy because physical execution is compared with a generated or selected reference.
- Contact, slip, penetration, continuity, reset, and COM measurements: simulation_only; surface contact is explicitly proxy-labeled.
- Semantic matching/diversity: self_supervised_proxy using the frozen released evaluator.
- V33 original-reference control: dataset-derived kinematic reference in one fixed simulator/controller; it is not physical ground truth or causal attribution.
- Human evaluation: none.

## Additional integrity findings

### Cached receipt and lineage enforcement: FAIL

Any existing receipt triggers certify_cached and returns after finding only a matching ledger label; evidence: scripts/physics_guided_queue.py:46-54. The active certifier states that it does not prove a missing historical binding existed at execution, and when lineage.json is absent it writes a new retrospective record; evidence: scripts/physics_guided_com_lineage_v24_20261012.py:44-45 and 149-158. The collector writes receipt.json before lineage.json, making that state reachable after interruption; evidence: scripts/physics_guided_collect.py:286-299. This contradicts docs/condition_feedback_formal_v28_20261012.md:23.

The flaw was not exercised in the bound result: all 74 raw directories contain lineage.json, every certification is bound_request_at_execution, every request/version hash matches, and no compatibility-amendment or lineage-history artifact is present. The current result is therefore corresponded, while the resumability control fails the protocol.

### Reset, frame, contact, COM, device, and trajectory binding: PASS

Reset capture and first-action checks are implemented in scripts/contact_sampling_collect_v5.py:13-41 and 55-81; the completion fence is simulate then fetch_results(True) at scripts/contact_sampling_collect_v8.py:12-29. The collector binds frame indices and post-physics state at scripts/physics_guided_collect.py:190-243 and 274-280. Native body COM is collected and used per environment/body at scripts/contact_point_collect_v20_20261012.py:14-20 and scripts/physics_guided_metric_v23_20261012.py:71-100,142-147. The CPU-native/CUDA-policy bridge and GPU PhysX route are explicit at scripts/contact_cpu_export_collect_v11.py:11-29 and scripts/closd_cpu_export_bridge.py:19-60.

All raw arrays are finite, reference_frame is exactly 0…119, valid masks are contiguous, all seven reset deltas are zero, completion fences exist, native contact and sensor hashes bind, 24-body COM records are present, and replay references match selected motions after the exact upstream float32 canonical transform. Maximum observed canonical error was 3.703056e-16 m, below the frozen 1e-12 bound.

### Endpoint timing event: WARN

The frozen protocol says position 10 cm, heading 10°, time 0.1 s and a four-second endpoint, but it does not define first arrival or entry into the position window; evidence: docs/physics_guided_20261011.md:7-9 and configs/physics_guided_20261011.json:17-29. V23 instead finds frames within 1 mm of the trajectory's own minimum target distance and chooses the one nearest the final frame; evidence: scripts/physics_guided_metric_v23_20261012.py:124-133. It gates that value at 0.1 s at :148-150.

This is material to interpretation. Of 1,177 replay rows whose final position is within 10 cm, 373 fail the frozen closest-approach timing gate. An audit-only first-entry calculation, using the same 10 cm and 0.1 s constants without changing historical scores, gives 325 final-position-and-first-entry-on-time cases versus 804 under the saved closest-approach event. Per-arm saved/first-entry counts are: raw 77/36, ordinary 61/29, random_best 79/32, native_single 51/17, native_best 90/38, proposed 120/49, no_feedback 51/17, shuffled 62/22, planned_prefix 126/54, tracking_only 87/31. This alternate calculation is diagnostic only and is not a retrospective replacement metric.

The ambiguity does not rescue C1: joint success remains zero under the frozen result, the numeric gain and confidence interval remain zero, and the progression gate remains false.

### Runtime witness root: WARN

runtime_witness.json is checked for status and exact-array booleans by scripts/run_condition_feedback_formal_v28_20261012.py:69-72 and 114-121, but its digest is not part of versions.json. The current witness hashes to sha256:ce07574de3dde3d800b4b6baa567f60b80131f6290fcc45558b6f1c74a9a6468 and all recorded checks are true. Future version roots should bind it directly.

## Supplemental diagnostics, kept outside C1 efficacy

### V33/V35 original-reference diagnostic: control verdict FAIL; current artifact correspondence PASS

The current V33 batches correspond: validation is 72/72 completed with one joint pass, test is 138/144 completed with zero joint passes, and no generation calls occurred; evidence: results/mocap_reference_control_v33_20261012/diagnostic_summary.json:1-102. V35 binds both raw packages, makes zero simulator/generator calls, and explicitly disclaims physical ground truth, causal attribution, efficacy, contact truth, and threshold revision; evidence: results/original_reference_attribution_v35_20261012/diagnostic_1791747761322343567.json:1-32.

The original V33 runner does not compare a reused validation receipt/output to witness-captured expected digests; evidence: scripts/run_mocap_reference_control_v33_20261012.py:54-77 and 94-106. Existing-output preparation verifies versioned source files but not frozen thresholds, while thresholds_changed is emitted as a literal; evidence: :26-37 and 107-110. It also consumes proposed_goal but does not assert proposed_goal == original_goal in the in-scope verifier. Current files do satisfy these equalities and match the witness, so no replacement was observed.

The V33 protocol document is stale: docs/mocap_reference_control_v33_20261012.md:19 still says the run is unexecuted, while queue_state.json:1-4 is COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC. This is a documentation WARN.

### V32 and repeatability: PASS as diagnostics only

V32 decomposes component passes and finds zero joint repairs; it does not establish causality or innovation. Evidence: results/condition_feedback_components_v32_20261012/diagnostic_1791747514277104976.json:3114-3310 and 4920-4927. Repeatability uses already-paid C1 executions, makes no simulation/generation calls, finds exact saved-array equality, and explicitly says it is not a fresh exact-input C2 experiment, accuracy evidence, physical feasibility evidence, or a quality-guard ablation; evidence: results/feedback_repeatability_diagnostic_20261012/diagnostic_1791747527722579420.json:1-35 and 294-305. C2 and C3 efficacy remain untested.

### V36 cache certificate: supplemental PASS for the present snapshot only

The V36 verifier hard-codes witness/runtime/terminal-summary anchors, checks witness inputs and sources, binds C1/V33 thresholds and manifests, verifies lineage requests and outputs, checks reset/reference clocks, and exactly recomputes all 216 stored V33 metric rows; evidence: scripts/verify_mocap_cache_binding_v36_20261012.py:14-18 and 64-132. Independent read-only rechecking found 69/69 input hashes equal, 216/216 metric rows exactly equal, identical C1/V33 thresholds, and exact proposed/original goal and heading equality in both batches. The certificate is at results/mocap_cache_binding_v36_20261012/certificate_1791749145717340182.json:1-105.

V36 is post hoc supplemental control evidence and is not integrated into the historical V33 runner. It does not erase the V33 control failure, change any score, or count as independent candidate-efficacy validation.

## Candidate efficacy and claim impact

- C1 joint-success efficacy: **unsupported**. Every arm is 0/432; gain is 0 percentage points; source-cluster 95% CI is [0,0]; each seed gain is zero.
- C1 endpoint and heading means: **supported descriptively in this frozen simulation**, with the closest-approach timing definition disclosed. They do not establish joint efficacy.
- C1 semantic/diversity gate: **supported as a released learned proxy only**; it cannot override the failed numeric gate.
- C2 efficacy and quality-guard ablation: **not tested**. V32/repeatability are diagnostic-only.
- C3 budget-allocation efficacy: **not tested**.
- Original-reference control: **supports a fixed tracker/representation/proxy confound in this setting only**. It does not prove generator-independent failure, motion infeasibility, or physical truth.
- Innovation: **not established**. The documentation correctly says so.
- G1, downstream training, robot validation, and hardware claims: **not run / unsupported**.

## Required actions

1. Preserve V28's failed gate and historical metrics. Do not relax thresholds, drop failures, or reinterpret the alternate timing diagnostic as the primary result.
2. Before any future reuse of the V28 cache, require lineage.json with certification == bound_request_at_execution and matching request/output hashes; never synthesize lineage from a receipt-only state.
3. For a future new experiment version, preregister the timing event: first entry into a fixed spatial window, closest approach, or another explicit event. Do not retrofit V28.
4. For future V33 reuse, use a separately bound fail-closed certificate such as V36, bind thresholds/witness digests in the run root, and assert proposed_goal/heading equality. Update the stale status prose without changing historical outputs.
5. Obtain an independent cross-family audit before treating this verdict as accepted rather than provisional.

