# Experiment Audit Report — V56/V57/V61 Integrity Delta

**Date:** 2026-10-12  
**Auditor:** GPT-5.6-Sol, ultra reasoning, fresh same-family read-only reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Scope:** V56 passive native-manifold collection, the V58/V60 failed continuations, the successful V61 continuation, and the one completed V57 CPU-only analysis. This report does not replace or revise the earlier V51 audit.  
**Overall verdict:** **WARN**

The result evidence is internally consistent and explicitly limited to simulator-only diagnosis. I found no fake real-world ground truth, no prediction-normalized acceptance score, no phantom result, and no dead diagnostic path. The warning is about reproducibility: the local checkout is at Git HEAD `f0e3c4014428dedcc3ce150e6069af4837149cd8`, the remote checkout is at `e1ef719d6702d15a5ab211db0e55be438b2d1c88`, and the authoritative remote V56/V57/V61 sources and results are untracked. File hashes bind the current evidence, but neither Git HEAD alone identifies the complete analyzed state.

## A. Ground-truth provenance: PASS

The evaluator has two distinct reference types, and both are labeled correctly:

- The replay tracking reference is the previously selected, model-generated and physics-refined V51 trajectory. It is explicitly recorded as `synthetic_proxy; prior model-generated physics-refined trajectory` in [continuation_manifest_v61.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/continuation_manifest_v61.json), JSON path `$.evaluation_reference_type`. The V56 documentation says the same at [native_manifold_v56_20261012.md](D:/具身智能/conditional-human-motion-generation/docs/native_manifold_v56_20261012.md:3).
- The contact manifold comes from Isaac Gym/PhysX `get_env_rigid_contacts`. It is simulator-native evidence, not dataset ground truth or real robot contact truth. The collection adapter labels it accordingly at [native_manifold_collect_v56_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/native_manifold_collect_v56_20261012.py:19), and the V57 analyzer repeats the limit at [analyze_native_manifold_v57_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:1) and [line 114](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:114).
- The official simulator-native contact API is available and used. Isaac Gym documents `RigidContact`, ground body index `-1`, local contact features, offsets, and contact-force magnitude at remote `/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym/docs/api/python/struct_py.html:816-900`.

Classification: primary `simulation_only`; replay tracking reference `synthetic_proxy`; no `real_gt`, `human_eval`, or real-force certification.

## B. Score normalization: PASS

V57 does not normalize an error or acceptance score by a prediction maximum, minimum, or mean. It compares speeds with the unchanged frozen threshold of `0.05 m/s` and reports raw counts, quantiles, and corresponding fractions. The calculation is visible at [analyze_native_manifold_v57_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:18), [line 56](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:56), and [lines 99-108](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:99). The threshold is bound in [frozen_thresholds.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/frozen_thresholds.json), JSON path `$.thresholds.sliding_speed_m_s`.

The `native_weighted_mean` divides by summed positive simulator contact-force weights as a descriptive weighted mean. It is not an acceptance score, and V57 also exposes unweighted point observations plus per-body-frame minimum and maximum speeds. The output explicitly records `metrics_replaced=false` and `thresholds_changed=false` at [diagnostic_1791771577179667295.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_analysis_v57_20261012/diagnostic_1791771577179667295.json), JSON paths `$.metrics_replaced` and `$.thresholds_changed`.

## C. Result existence and lineage: WARN

All required artifacts exist. The V57 diagnostic exists locally and remotely with SHA-256 `621ce100ab331637854571dc8bd74b99571bb9ed326cdf319f62220f849308df`. Its 13 recorded input hashes and its source hash were independently compared with the current authoritative remote files; every value matches.

The terminal V56/V61 state is real:

- [state.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/state.json), `$.status`, is `COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS`; `$.continuation` is `V61`.
- [summary.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/summary.json), `$.arms.feedback` and `$.arms.random`, contains 72 cases per arm, all 33 exact-array comparisons true, exact frozen metrics, `288602` and `288411` native records, and zero parser errors.
- [budget_ledger.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/budget_ledger.json) has exactly two successful `measurement` rows: feedback `81.41063746996224 s` and random `81.37118211574852 s`, total `162.78181958571076 s`.
- [continuation_receipt_v61.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/continuation_receipt_v61.json), `$.status` and `$.exit_code`, records `COMPLETE` and `0`; `$.simulation_batches_this_invocation` is `1`.
- The independent V61 witness exists locally and remotely with SHA-256 `315332419cb45fa4e6dc2f066c1e163e2f9ef1618fa717221541e2edc2da80d2` and confirms the same counts at [CONTINUATION_DOCUMENT_WITNESS_V61.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/CONTINUATION_DOCUMENT_WITNESS_V61.json), JSON paths `$.ledger_entry_count`, `$.all_random_exact_arrays`, and `$.random_native_records`.

The warning does not invalidate the current hash-bound output. It means the run cannot be reconstructed from a Git commit alone until the untracked remote files are committed or archived under an immutable identifier and the local/remote HEAD difference is documented.

## D. Dead-code detection: PASS

The measurement and analysis functions are live:

- V56 injects `read_contacts` immediately before saving the same-step state snapshot at [native_manifold_collect_v56_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/native_manifold_collect_v56_20261012.py:13).
- The runner executes the replay collector, checks zero generation calls, validates the native file, compares all saved arrays, and writes the equality record at [run_native_manifold_v56_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/run_native_manifold_v56_20261012.py:64) through [line 94](D:/具身智能/conditional-human-motion-generation/scripts/run_native_manifold_v56_20261012.py:94).
- V61 uses the same executor for the missing random batch and writes the terminal summary and receipt at [continue_native_manifold_v61_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/continue_native_manifold_v61_20261012.py:50) through [line 66](D:/具身智能/conditional-human-motion-generation/scripts/continue_native_manifold_v61_20261012.py:66).
- V57 calls its quantile/count helpers and emits per-case plus aggregate native/proxy statistics at [analyze_native_manifold_v57_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:18) and [lines 33-118](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:33).

V58 and V60 are not silently counted as successful code paths. Their failed preflights and stack traces remain in [CONTINUATION_DOCUMENT_WITNESS_V58.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/CONTINUATION_DOCUMENT_WITNESS_V58.json), JSON paths `$.status` and `$.combined_output`, and [CONTINUATION_DOCUMENT_WITNESS_V60.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/CONTINUATION_DOCUMENT_WITNESS_V60.json), the same paths.

## E. Scope: PASS

The actual and stated scopes agree:

- validation split only;
- one seed, `11`;
- 24 source IDs per arm;
- three target offsets per source, `-0.15`, `0`, and `+0.15 m`, each represented 24 times;
- 72 cases per arm, with the same case IDs in both arms;
- 36 dance and 36 jump cases per arm;
- two fixed replay arms, feedback and random;
- zero generator calls;
- all 72 replays in each arm completed, while original `joint_pass` is `0/72` for each arm.

The frozen declaration is in [versions.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/versions.json), JSON paths `$.split`, `$.seed`, `$.cases_per_batch`, `$.sources`, `$.expected_replay_batches`, and `$.generation_calls`. The output cases confirm the complete source/offset cross-product at [diagnostic_1791771577179667295.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_analysis_v57_20261012/diagnostic_1791771577179667295.json), JSON paths `$.arms.feedback.cases` and `$.arms.random.cases`.

The language does not exceed the evidence. [native_manifold_continuation_v61_20261012.md](D:/具身智能/conditional-human-motion-generation/docs/native_manifold_continuation_v61_20261012.md:13) and the diagnostic `$.scope` explicitly rule out efficacy, innovation, real contact ground truth, and broader controller-independent conclusions.

## F. Evaluation type: PASS

| Component | Classification | Basis |
|---|---|---|
| Native contact manifold | `simulation_only` diagnostic | Isaac Gym/PhysX contact records |
| Tracking comparison | `synthetic_proxy` | selected model-generated, physics-refined replay trajectory |
| Frozen surface-slip metric | `simulation_only` proxy | collision-primitive lowest surface point plus simulator state/contact-force proxy |
| Real contact or force truth | absent | explicitly disclaimed |
| Human evaluation | absent | no human judgments |

This matches [summary.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/summary.json), JSON paths `$.arms.feedback.evaluation_type` and `$.arms.random.evaluation_type`, and the V57 diagnostic `$.evaluation_type`.

## Contact point, frame, COM, and timing audit

**Ground/body side: PASS.** V57 treats `body0 == -1` or `body1 == -1` as the ground side and uses the other body's local contact data at [analyze_native_manifold_v57_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:44). This matches Isaac Gym's documented `-1` ground convention.

**Contact point: PASS for this output, with a future-use limit.** V57 uses `local_pos + offset` at [line 52](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:52). Isaac Gym describes `local_pos` as the feature where normal force is applied and `offset` as the shift to the friction surface. I independently scanned both full native-contact JSON files: every `offset0` and `offset1` component is exactly zero across all `288602 + 288411` records. Therefore normal-feature and friction-surface points coincide numerically here. If later runs contain nonzero offsets, these two points should be reported separately.

**COM correction: PASS.** V57 computes `v_point = v_COM + omega × (r_point - r_COM)` at [lines 53-55](D:/具身智能/conditional-human-motion-generation/scripts/analyze_native_manifold_v57_20261012.py:53), using each environment/body's simulator COM. The frozen V23 proxy uses the same convention at [physics_guided_metric_v23_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/physics_guided_metric_v23_20261012.py:82), and the articulated calibration records its limited validation at [articulated_contact_fixture_v22_20261012.py](D:/具身智能/conditional-human-motion-generation/scripts/articulated_contact_fixture_v22_20261012.py:68).

**Time alignment: PASS.** The V56 adapter queries contacts after the wrapped reward computation and immediately before the same-step position/quaternion/linear/angular state copies. The surrounding state capture is at [physics_guided_collect.py](D:/具身智能/conditional-human-motion-generation/scripts/physics_guided_collect.py:213) through [line 238](D:/具身智能/conditional-human-motion-generation/scripts/physics_guided_collect.py:238). Raw records cover steps `0-119` and environments `0-71` in both arms.

**Lambda: PASS as a simulator diagnostic.** Isaac Gym calls `lambda` contact-force magnitude. V57 filters to positive values and uses them only as weights in one descriptive statistic. The docs already prohibit interpreting those values as real forces at [native_manifold_v56_20261012.md](D:/具身智能/conditional-human-motion-generation/docs/native_manifold_v56_20261012.md:17).

## Verified V57 values

| Quantity | Feedback | Random |
|---|---:|---:|
| Cases / sources | 72 / 24 | 72 / 24 |
| Native point observations | 24,095 | 24,058 |
| Matched support body-frames | 9,051 | 9,099 |
| Positive native ground body-frames | 9,216 | 9,257 |
| Frozen proxy above 0.05 m/s | 4,717 | 4,729 |
| All native contact points above 0.05 m/s | 4,470 | 4,499 |
| At least one native point above 0.05 m/s | 5,029 | 5,084 |
| Proxy above, all native points at/below | 11 | 5 |
| Proxy support with no positive native ground contact | 2 | 3 |
| Fraction with all native points above threshold | 0.493868 | 0.494450 |

These values match [diagnostic_1791771577179667295.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_analysis_v57_20261012/diagnostic_1791771577179667295.json), JSON paths `$.arms.feedback` and `$.arms.random`, and the public narrative numbers at [candidate_mechanisms_20261011.md](D:/具身智能/conditional-human-motion-generation/docs/candidate_mechanisms_20261011.md:11). The body-frame and point observations are dependent within source and frame; they are not independent trials.

## Preserved failures

- The original V56 `--run` attempt failed before starting the random arm because the expected copied witness was absent. Its generic `FileNotFoundError` remains in [failed_main_invocation_state.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/failed_main_invocation_state.json), JSON path `$`.
- V58 failed its preflight assertion before random replay. The exact stack trace and `random_batch_started=false` remain in its witness.
- V60 failed its preflight because transformed code passed a string to a hash helper requiring a `Path`. The exact stack trace and `random_batch_started=false` remain in its witness.
- V61 is a separate, single successful continuation that ran the one missing random replay. Its manifest binds all earlier failure records at [continuation_manifest_v61.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_v56_20261012/continuation_manifest_v61.json), JSON paths `$.preserved_prebatch_failures` and `$.original_failed_invocations_not_success`.

## Claim impact

- **Supported within this two-batch scope:** passive native-contact instrumentation left all 33 stored trajectory, observation, initial-state, and sensor arrays unchanged and exactly reproduced the frozen metrics.
- **Supported with simulator-only, one-seed qualification:** choosing the frozen proxy's lowest collision-primitive point does not by itself explain most threshold exceedances in the matched support observations. Only `11/9051` feedback and `5/9099` random body-frames had proxy speed above threshold while every reported native point stayed at or below it.
- **Unsupported:** feedback improves contact kinematics over random selection. V57 ran no source-level inferential efficacy comparison, and both arms have `0/72` original joint passes.
- **Unsupported:** real robot feasibility, real contact/force truth, force-control validity, or controller-independent physical validity.
- **Unsupported:** innovation acceptance or any change to frozen V51 scores, thresholds, or acceptance status.
- **False if stated:** V56, V58, and V60 succeeded. Their failures are retained; only V61 completed the missing arm.

## Action items

1. Commit or immutably archive the exact V56/V57/V61 sources and audit-relevant artifacts, then record the authoritative identifier. Reconcile or explicitly document the local/remote HEAD divergence.
2. Keep the measured point labeled as a simulator friction-surface material point. For future nonzero offsets, report normal-feature and friction-surface velocities separately.
3. Keep raw counts beside fractions. Any comparative inference must aggregate or resample at the original-source level and use additional seeds or a held-out test protocol.
4. Continue to cite V56/V58/V60 as retained failures and V61 as a distinct continuation.
5. Do not use this diagnosis as evidence of method efficacy, real-world contact truth, force validation, controller-independent feasibility, or innovation acceptance.

## Primary hashes

- V57 diagnostic: `621ce100ab331637854571dc8bd74b99571bb9ed326cdf319f62220f849308df`
- V57 source: `cd2fb7bafa7f85ee7ac1f903d7e7439accaccb23d9997b14488f37db184fe49b`
- V56/V61 summary: `c1c0f3e05f0973fda51e693dde94840d1912e6d4dcd5ae32c1332ea98261846e`
- V56/V61 budget ledger: `a4f027f83a7c10fe5fffb27229d78aac9988a3778becfd82bda43b9e2ce99c0a`
- V61 manifest: `66dd45e8e792faade8be526e450c52c68e6d7ce4dac4563cab10a262212c08d0`
- V61 receipt: `24a9269e305e46de378d96660d2abc9388fcaca02d2a88a376997e1742f23725`
- V61 independent witness: `315332419cb45fa4e6dc2f066c1e163e2f9ef1618fa717221541e2edc2da80d2`
- Feedback native contacts: `a95dcf7b19b7f01272d5ef232d76c2a07bd1f3da4f45fdca8991d0610a612a70`
- Random native contacts: `97dea899b1435e818d61b416b772cff0f41e73fe8d02aeecceb552ca2c8e2f35`

The full audited-input hash map is in [EXPERIMENT_AUDIT.json](D:/具身智能/conditional-human-motion-generation/results/native_manifold_analysis_v57_20261012/EXPERIMENT_AUDIT.json), JSON path `$.audited_input_hashes`.
