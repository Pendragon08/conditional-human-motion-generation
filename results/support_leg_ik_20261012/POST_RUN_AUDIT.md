# Post-run integrity audit: bounded support-leg IK diagnosis

**Status: WARN · provisional · same-family independent review**

All 216 classifications, frozen analysis metrics, source-cluster bootstrap results, reference invariants, exported state checks, component counts, real costs, and outcome bindings reproduce. The audit is `WARN` because the child collector did not execution-time re-hash the loaded replay content or query the actual upstream checkout state before writing `bound_request_at_execution`. The evidence also supports only a one-seed internal simulation/proxy engineering diagnosis.

The frozen decision is reproduced: `STOP_THIS_LIMITED_LEG_TRANSLATION_AND_POSITION_IK_PARAMETERIZATION`. Both arms have `0.0` percentage-point joint-pass gain and a 24-source cluster-bootstrap interval of `[0.0, 0.0]`. That degenerate interval describes this observed slice; it does not prove population equivalence.

## A–F integrity checks

| Check | Status | Finding |
|---|---:|---|
| **A. Ground-truth provenance and proxy labeling** | **PASS** | Endpoint goals are derived from the fixed dataset motion before generation. Replay tracking uses a prior simulated/model trajectory and is correctly labeled synthetic-proxy tracking. Surface slip is explicitly a collision-geometry/body-force proxy, without a real contact-point or real-world GT claim. |
| **B. Normalization and denominators** | **PASS** | All gates use frozen absolute SI thresholds. No metric divides by its own prediction maximum, minimum, or mean. Every reported scalar has 72 finite values and zero missing/nonfinite values in each arm. The component projection remains in m/s; the pooled slip ratio has an explicit pooled support-observation denominator and is separate from the 72-case mean. |
| **C. Files, receipts, terminal scope, counts, and numeric match** | **WARN** | Current requests, output hashes, receipts, logs, counts, classifications, and frozen analysis values match. The actual SSH run and analyzer were observed at exit 0. The child did not recompute `replay_motion` after loading the file, and runtime `source_commit` was copied from configuration rather than queried from the loaded checkout. The external terminal record is not an autonomous server parent-exit receipt. |
| **D. Metric implementation actually called** | **PASS** | The V43 adapter chain patches the effective collector to `physics_guided_metric_v23.metrics`; the collector calls it per case. The frozen analyzer imports the same implementation and recomputes all 72 rows per arm. The component diagnostic independently reconstructs the slip counts. |
| **E. Sources, seeds, reference locks, state, budget, bootstrap, and claim scope** | **WARN** | The fixed source is 24 clusters × 3 offsets, seed 11. Prefix/endpoints/targets are unchanged and the two-link IK geometry passes float32 invariants. Bootstrap is source-clustered. The study mixes upstream train/test provenance, has one seed, lacks an untouched-test/pretraining-exposure claim, and cannot establish hidden PhysX state identity. A summary field also names the current append-only parent ledger as if it were the historical baseline-only snapshot. |
| **F. Evaluation type** | **PASS** | `simulation_only; synthetic_proxy tracking`, plus a read-only post-hoc proxy-component decomposition. The decomposition is not an alternate threshold, alternate score, simulator experiment, causal test, or efficacy acceptance. |

## Reproduced outcome

| Arm | Completed | Condition pass | Physical pass | Tracking pass | Joint pass |
|---|---:|---:|---:|---:|---:|
| Cached baseline | 72 | 5 | 0 | 68 | 0 |
| Cartesian leg translation | 72 | 7 | 0 | 68 | 0 |
| Two-link positional IK | 72 | 6 | 0 | 68 | 0 |

The independently recomputed means are:

| Arm | Endpoint m | Heading rad | Time s | Mean per-case slip proxy | Penetration | Jerk p95 m/s³ | Tracking mean m | Tracking p95 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.195906993 | 0.212179964 | 0.583333364 | 0.542411146 | 0 | 1611.056874 | 0.055011089 | 0.087769335 |
| Cartesian | 0.193637800 | 0.206981297 | 0.587962994 | 0.548271445 | 0 | 1591.930024 | 0.055845583 | 0.089653040 |
| Two-link IK | 0.196799634 | 0.202363607 | 0.569907437 | 0.544229579 | 0 | 1585.041459 | 0.055898012 | 0.090405747 |

All acceptance classifications and frozen analysis metrics agree under the declared numeric comparison. Exact cross-runtime arithmetic retained three low-order differences: one analysis mean at `2.7755575615628914e-17`, component weighted means at up to `2.7989e-11 m/s`, and one descriptive `surface_slip_proxy_mean_m_s` at up to `6.54324e-10 m/s`. They change no support/slip count, slip fraction, gate, classification, frozen decision, or permitted claim.

## Proxy-component audit

The pooled numerator counts supported body-frame observations whose material-point horizontal speed exceeds `0.05 m/s`; the denominator is every supported body-frame observation. These pooled ratios differ legitimately from the analysis file’s equally weighted mean of 72 per-case ratios.

| Arm | Slip / support observations | Pooled fraction | COM-linear m/s | Rotational-term m/s | Material m/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 4650 / 8927 | 0.520891677 | 0.171209415 | 0.151612180 | 0.191771760 |
| Cartesian | 4740 / 8990 | 0.527252503 | 0.172024196 | 0.150942792 | 0.193410746 |
| Two-link IK | 4715 / 8972 | 0.525523852 | 0.169096519 | 0.152285002 | 0.189322542 |

Term magnitudes are support-weighted and are not additive. Signed projections may cancel. This diagnosis is descriptive and noncausal; it is not a replacement score or threshold.

## Reference, state, and source invariants

- All references have shape `120 × 72 × 24 × 3`, dtype `float32`, and finite values.
- Frames 0–60, frame 119, goals, headings, and non-leg coordinates retain the frozen values. All 72 selected inputs were complete; the incomplete-source preservation branch was not exercised.
- The maximum requested horizontal shift is `0.07820509374 m`. The Cartesian ablation changes segment length by up to `0.0579819083 m`; the IK arm reduces maximum segment error to `8.9406967e-08 m` and moves the knee by at most `0.0596136041 m`.
- Exported initial-state archives agree across arms. Seven bootstrap tensors equal their first-action tensors, and all four recorded rigid-cache differences are zero. Hidden PhysX engine state identity is not established.
- The source slice contains 24 clusters: 19 official train-derived and 5 official test-derived; equivalently, 57 and 15 inputs. This is internal validation, not an untouched test set, and pretrained exposure was not independently established.
- The controller and generator hashes are respectively `7ead3778…` and `3ca68ae5…`. All versioned source, manifest, threshold, analysis-source, raw-output, receipt, and current analysis/component bindings checked by this audit match.

The native CLoSD implementation boundary was checked read-only at SHA-256 `dff1ca9bfb15c3058914c326cba83a07f69cd91952cd4e03ecfe1ad3d391e394`. The frozen interface consumes position and linear-velocity reference observations; adding foot quaternion, force, or contact fields alone would not establish that the frozen policy executes those constraints.

## Receipts, costs, and resource bounds

The run observation records SSH exit 0, chunk `5faa0c`, and sentinel `SUPPORT_LEG_IK_ADJUDICATED_COMPLETE`. The analyzer observation records exit 0 and chunk `028dc7`. `EXECUTOR_TERMINAL_OBSERVATION.json` is accurately labeled as an external tool observation. `pipeline_finally_receipt.json` is a process-finally record; neither is presented as an autonomous server parent-exit receipt.

Real charged costs are `49.1905804593 s` for source construction, `50.4387053065 s` for the cached baseline’s original charge, `49.7906149086 s` for Cartesian, and `49.8940196168 s` for IK. The two new arms total `99.6846345253 s`; saved-array numerical adjudication costs `0.90373248 s`.

The historical baseline-only ledger hash begins `d450a9…`. The current append-only three-row parent ledger begins `c1a19…`. Every row and cost matches, but `summary.json.original_baseline_budget_ledger_sha256` points to the current ledger, so the field name is misleading. It should be clarified prospectively without rewriting this run.

The resource snapshot is `CAPS_PASS`: total GPU-job wall time `4.56358397 h`, comparison time `2.12490853 h`, output `9,014,172,435 B`, and free space `36,603,215,872 B`. Historical failed attempts remain in the resource accounting.

## Historical FAILs and checker adjudication

The original document witness remains `FAIL` with 117 scalar checks. Those checks required exact equality between float64 logged norms and norms recomputed from saved float32 reference vectors. The independent scalar contract verifies the saved float32 arrays and documented geometric bounds; this audit does not impose undocumented exact logged-norm equality.

The continuation witness remains `FAIL` for one whole-ledger equality assertion. The independent addendum reconstructs the historical baseline-only snapshot and separately records the later append-only ledger. This audit does not impose historical full-ledger equality.

The new read-only audit checkers also keep their literal `FAIL`/`FAIL_OR_DIVERGENCE` labels. Their extra assertions used a file SHA in place of the documented NPZ-content hash, expected replay mode for the native V51 source construction, or required cross-runtime bitwise equality for float32 canonicalization, velocity arithmetic, solver recreation, and one descriptive reduction. The full code-backed treatment is in `.aris/traces/experiment-audit/leg_ik_20261012/checker_adjudication.json`. The real child-side replay and upstream-state binding warnings remain open for this historical run.

## Claim ceiling and actions

The permitted claim is:

> In this one-seed internal simulation replay over 24 source clusters, neither bounded Cartesian leg translation nor the tested classical two-link positional IK changed joint-pass rate from zero. This supports stopping only the tested parameterization and records a descriptive proxy decomposition.

Do not infer population equivalence, general ineffectiveness of IK, real contact-point slip, torque/collision/full-dynamics correctness, candidate efficacy, confirmed innovation, novelty, G1 performance, downstream training benefit, deployment behavior, or untouched-test generalization.

For future confirmatory work:

1. Recompute replay-content and versions hashes inside the child after loading inputs; bind actual upstream HEAD/dirty state and model-weight hashes per batch. Do not retroactively certify this run.
2. Preregister multiple seeds, genuinely held-out sources with pretraining-exposure accounting, and an independent or real contact evaluator.
3. Give the historical baseline-only snapshot and current append-only ledger distinct field names.
4. Keep pooled component statistics separate from the per-case frozen score and label them descriptive/noncausal.
5. Preserve both historical FAIL witnesses, their independent addenda, and the external-observation label.

This review ran two separately named CPU-only audit scripts and performed no simulation, training, generation, installation, experiment retry, target change, threshold change, source rewrite, or result rewrite.

