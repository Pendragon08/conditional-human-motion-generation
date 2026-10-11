# Experiment Audit Report

**Date:** 2026-10-12  
**Auditor:** fresh Codex read-only integrity reviewer; same-family  
**Acceptance:** provisional  
**Project:** V24 COM validation, V26 numeric-runtime repair, and V27 fixed-reference replay  
**Repository snapshot:** `02a2ed646abf7e0eb7dca636e994124d3f91dcae`  
**Overall verdict:** **WARN**  
**Integrity status:** **warn**

The archived V24/V26/V27 evidence is internally coherent. Independent parsing reproduced every reported count and all 576 V24-plus-V27 pass classifications. The fixed original targets, explicit reference clock, retained failures, frozen thresholds, replay inputs, and same-configuration bindings check out. No prediction-derived score normalization, missing result, altered failure denominator, or unlabelled contact ground truth was found.

The warning is about verification and reporting code, not a mismatch in the current numbers. The report builder declares a fixed-replay pass whenever the replay JSON is merely present and truthy; the V24 cached-result path returns before running several new-batch metadata guards; and several metric/helper definitions are not live in this pipeline. The evidence therefore supports a narrow simulation diagnostic, with same-family provisional acceptance. It does not establish true contact physics, candidate efficacy, generalization, or real-world validity.

## Supplemental correction: config-digest representations

An earlier version of this report incorrectly called the threshold file's `config_sha256=f7e06e...` stale. That finding is **retracted**. The superseded report versions had SHA-256 `d590ff100d6025be011b3728a2f91cdabf1ceac97ecd6003fcbfa9c03703346e` (Markdown) and `34fcaa91f580b165351228121895cd6669b14f2b41cd777e5bcda1965f5acd28` (JSON).

Independent recomputation over the same config produced two deliberately different digests: raw file bytes hash to `2d8fd0e49ce7ed37393e1962dcd711b5a6e8f26de7e0ca4ef48964a6fd081502`, while `sha256(json.dumps(json.loads(file), sort_keys=True).encode())` hashes to `f7e06ee902a61f7362a086ca6c5d5969cec258b0125edda4205157adaddf48cd`. `physics_guided_queue.py` creates and asserts the latter parsed/canonical JSON digest (`scripts/physics_guided_queue.py:185-193`); requests and versions bind the former raw-byte digest (`scripts/physics_guided_com_lineage_v24_20261012.py:28-34`; `results/contact_com_validation_v24_20261012/versions.json:46`). Both correctly identify the same config under different declared representations. Same-configuration evidence therefore **passes**. The overall WARN remains for the independent report-gate, cached-guard, and dead-code findings.

## Audit boundary and concurrent input change

No simulator, collector, witness, verifier, report builder, or experiment command was run. The audit only read local artifacts and, for arrays omitted from the public export, used read-only SSH inspection. No implementation file was edited.

Three report-facing files changed while the audit was in progress, in commit `02a2ed646abf7e0eb7dca636e994124d3f91dcae`. The change added a V28 queue-state progress snapshot. The current V24/V27 metric, collector, summary, and replay files did not change. This report anchors its findings to the final on-disk hashes below. The V28 launch/progress claims in `README.md:3`, `scripts/build_contact_measurement_report_20261011.py:33,54,58,100-101`, and `results/contact_measurement_review_20261011/measurement_review.html:5` are outside this audit; a queue-state snapshot is not treated here as a completed result or live-process attestation.

| File | Initial SHA-256 at audit start | Final SHA-256 audited |
|---|---|---|
| `scripts/build_contact_measurement_report_20261011.py` | `c283de078dd038990a385abbdce99d5a83cd46f8efad075847afd4de4733b4c8` | `a1de5cf5724ccff5d4296c108c1fbc0d5cbb5261b45764dbe77a43c3313017d2` |
| `README.md` | `8a557c70a8c9ae67a4445b2d2963958ac4028a5cf081f8e5bb26c1e2c998a0ca` | `da1577e5d5c69dd4807ff6f69fc7a73e7ddd48f23beb76f053083bc73ddfd5be` |
| `results/contact_measurement_review_20261011/measurement_review.html` | `50306bc3a53a4bd3524a1e4f4e2e9a3774b5703954e7e6fcd431bc19ab5475d9` | `6a914159ba970e0c2f5ff5d3e24f6526691ead510ce0baebd9561122fdc5208e` |

After the audit verdict was formed, `scripts/build_contact_measurement_report_20261011.py` changed again in the shared working tree from audited SHA-256 `a1de5cf5...` to `bfadc5761d74d8a46fd3bd29a7e302676f25c3cf8e89446b491c8ae4eed94682` as a separate report-gate remediation. That later version is not evaluated here and does not replace the exact audited-input hash. No bound V24/V27 metric, collector, trajectory, or summary input changed with it.

## Check results

### A. Ground-truth and reference provenance: PASS, bounded

The endpoint position and heading are decoded from dataset motion, frozen before generation, and offset by the predeclared horizontal case offset (`scripts/physics_guided_collect.py:94-106,129-138`). V23 scores `original_goal` and `original_heading`, not proposed or generated targets (`scripts/physics_guided_metric_v23_20261012.py:103-123`). The public artifacts share original-goal SHA-256 `d008fae01d78f5f8f752aa4791b824540aec5791fd36173306881bcb09584809` (`results/contact_com_validation_v24_20261012/public/raw/val_planned_s11/metadata.json:2125`; `results/fixed_reference_com_v27_20261012/selected/raw_reference.json:3605`).

The tracking reference is not real ground truth after the 60-frame context. V24 tracks a frozen model-generated plan, while V27 replays either that plan or the frozen simulated body trajectory selected from the same V24 batch (`scripts/physics_guided_collect.py:190-204`; `scripts/run_fixed_reference_com_v27_20261012.py:63-71`; `scripts/physics_guided_queue.py:138-166`). Those are `synthetic_proxy` references inside a simulated environment.

The foot measurement is explicitly a calibrated proxy. It computes a COM-corrected surface-point velocity and gates support using body net vertical force plus collision-surface height (`scripts/physics_guided_metric_v23_20261012.py:71-100`). Native contact manifolds and passive foot-sensor outputs are retained as diagnostics but are not score inputs (`results/fixed_reference_com_v27_20261012/public/raw/val_raw_replay_s11/metadata.json:3818-3827`). V27 expressly states that it is not ground-truth contact force or arbitrary rolling-contact truth (`results/fixed_reference_com_v27_20261012/replay_summary.json:31-32`). This provenance is correctly disclosed, so the check passes; it places a strict ceiling on the claim.

The calibration boundary is also explicit in its source evidence. V19 distinguishes COM-referenced from origin-referenced rigid-body velocity on an isolated rotating body, with maximum errors `7.0334e-05` versus `0.18006` m/s, and says it is not articulated contact-slip calibration (`results/velocity_reference_fixture_v19_20261012/first_attempt/summary.json:2-10,31-36`). V22 checks one known translational articulated static/slide/airborne fixture and reports the expected proxy values, while excluding arbitrary rolling contacts and real forces (`results/articulated_contact_fixture_v22_20261012/first_attempt/summary.json:2-40,90-94`). These controls support the stated convention and limited proxy behavior only.

The custom source split is derived from the project manifest, not an official unseen benchmark evaluator. Source selection says it does not claim pretraining-data independence (`scripts/physics_guided_prepare.py:69-96`). The 24 V24/V27 sources contain 19 official-train and 5 official-test source labels. They are project validation inputs, not an exclusively official-test set.

### B. Score normalization: PASS

Input features use pinned dataset statistics, `(raw - mean) / std` (`scripts/physics_guided_collect.py:56-63`). Reported endpoint, heading, timing, tracking, jerk, slip, penetration, and airborne values remain SI quantities or direct fractions. Pass/fail uses fixed thresholds (`scripts/physics_guided_metric_v23_20261012.py:148-160`; `results/contact_com_validation_v24_20261012/frozen_thresholds.json:2-17`). No reported value is divided by the evaluated prediction's own maximum, minimum, or mean.

`rank_candidate` divides endpoint and heading errors by fixed thresholds for candidate ordering (`scripts/physics_guided_metric_v23_20261012.py:164-171`), but V23's function is not called by this V24/V27 path. V27's selector imports the historical ranking helper and has exactly one construction candidate per case (`scripts/physics_guided_queue.py:138-155`; `results/fixed_reference_com_v27_20261012/selected/raw_reference.json:3606`). No normalized ranking score is reported as an evaluation result.

### C. Result files, numbers, lineage, and denominators: WARN

All requested result files exist. Independent aggregation reproduced the following values from the public per-case metric rows, and every recorded public metric/metadata SHA in the summaries matches its file.

| Evaluation | Attempted | Completed | Condition | Physical proxy | Tracking | Joint | Terminated |
|---|---:|---:|---:|---:|---:|---:|---:|
| V24 planned seed 11 | 72 | 70 | 23 | 0 | 53 | 0 | 2 |
| V24 native seed 11 | 72 | 70 | 13 | 0 | 52 | 0 | 2 |
| V24 planned seed 12 | 72 | 70 | 23 | 0 | 53 | 0 | 2 |
| V24 native seed 12 | 72 | 70 | 18 | 0 | 54 | 0 | 2 |
| V24 planned seed 13 | 72 | 70 | 21 | 0 | 52 | 0 | 2 |
| V24 native seed 13 | 72 | 69 | 16 | 1 | 54 | 1 | 3 |
| **V24 total** | **432** | **419** | **114** | **1** | **318** | **1** | **13** |
| V27 raw-reference replay | 72 | 71 | 23 | 0 | 54 | 0 | 1 replay / 2 construction |
| V27 refined-reference replay | 72 | 70 | 14 | 0 | 68 | 0 | 2 replay / 2 construction |

V24's six rows are stored at `results/contact_com_validation_v24_20261012/validation_summary.json:4-109`, with the 432-attempt denominator and diagnostic qualification at `:112-119`. The quality diagnosis preserves twelve 36-attempt seed/mode/family cells and totals 1/432 slip-proxy passes, 432/432 penetration passes, and 151/432 continuity passes (`results/contact_com_validation_v24_20261012/quality_diagnosis.json:8-637`). All 13 early terminations remain in the 432 denominator. V23 defines completion as exactly 120 valid frames with no termination (`scripts/physics_guided_metric_v23_20261012.py:103-117`).

V27 retains all 72 choices in each arm. Padded construction failures cannot become combined successes because combined success requires both replay `joint_pass` and `construction_complete` (`scripts/run_fixed_reference_com_v27_20261012.py:73-88`; `scripts/physics_guided_queue.py:146-151`). The stored V27 counts and narrow scope match `results/fixed_reference_com_v27_20261012/replay_summary.json:2-40`. `full_physical_quality_floor=true` means both replay arms have zero physical passes; it is not a quality-pass flag (`scripts/run_fixed_reference_com_v27_20261012.py:87`).

All 576 V24/V27 rows' `condition_pass`, `physical_pass`, `tracking_pass`, and `joint_pass` flags agree with the frozen numeric criteria. V26 records NumPy 1.23.5, eleven unchanged preserved outputs, batch count 1 before and after, no cached resimulation, and unchanged thresholds (`results/contact_com_runtime_v26_20261012/runtime_check.json:2-29`). This certifies numeric-runtime repair and cached-output reproduction; it is not evidence of contact truth.

Two code-level weaknesses make this check WARN despite the current match:

1. `measurement_gate='CALIBRATED_PROXY_AND_FIXED_REPLAY_PASS'` is selected whenever `replay27` is a truthy JSON object. The builder does not require `status == COMPLETE_FIXED_REFERENCE_CALIBRATION`, validate the entry counts, or inspect verification flags before emitting the pass text (`scripts/build_contact_measurement_report_20261011.py:29-63,95-103`). The current replay file is complete, so the present HTML statement is factually consistent; the gate itself is brittle and could promote a malformed or failed but truthy file.
2. A cached V24 receipt is certified and returned at `scripts/run_contact_com_validation_v24_20261012.py:90-93`, before the new-batch completion-fence, sensor, CPU-export, native-contact, and reset assertions at `:128-139`. Those fields were checked directly in the current metadata, but the cached path does not enforce the same guards.
The earlier config-hash warning was a reviewer error, corrected above. The parsed/canonical digest `f7e06e...` is created and asserted by the queue; the raw-byte digest `2d8fd0...` is separately bound by requests and versions. Their difference is representational, not a config mismatch.

### D. Dead code and non-gating metrics: WARN

The live metric path calls V23 `metrics` and `surface_metrics`, and their outputs appear in the per-case results (`scripts/physics_guided_collect.py:274-299`; `scripts/physics_guided_metric_v23_20261012.py:71-160`). `surface_slip_proxy_mean_m_s`, `penetration_depth_p95_m`, `airborne_fraction`, and `support_observations` are diagnostic values; the physical gate uses slip fraction, penetration fraction, and jerk only (`scripts/physics_guided_metric_v23_20261012.py:93-100,148-160`). Their presence is not evidence that they gate acceptance.

V23's `bounded_update`, `rank_candidate`, and `unit_witness` are not live in this V24/V27 result path. The selector imports historical `physics_guided_core.rank_candidate`, and V27 supplies only one candidate. V24 also imports `correction`, `select`, and `load` without calling them (`scripts/run_contact_com_validation_v24_20261012.py:12`). Native-contact and passive-sensor channels are collected but deliberately excluded from scoring. These do not fabricate current numbers, but the unused definitions/imports and diagnostic-only fields should not be described as exercised acceptance logic.

### E. Scope assessment: PASS with a strict claim ceiling

V24 contains 24 distinct sources, three offsets, three **generation-noise** seeds, and two prefix modes: 432 attempts, not 432 independent sources (`configs/physics_guided_20261011.json:5-11`; `scripts/run_contact_com_validation_v24_20261012.py:144-167`). The simulation environment seed and top-level seed remain 11; the recorded `seed` 11/12/13 changes diffusion noise only (`scripts/reset_aligned_collect_v4.py:1-5,42-48,82-83`; `scripts/physics_guided_collect.py:167-175`; representative `results/contact_com_validation_v24_20261012/public/raw/val_native_s13/metadata.json:2780,2790-2794`). The README correctly says three generation seeds.

V27 uses the same 24 sources times three offsets, generation seed 11, one construction candidate, and the single V24 planned-seed-11 batch for both raw and refined replay construction (`scripts/run_fixed_reference_com_v27_20261012.py:63-88`). Its independent document witness explicitly covers raw replay only and excludes refined replay, physical quality, efficacy, and the full formal comparison (`results/fixed_reference_com_v27_20261012/DOCUMENT_WITNESS.md:9-11`). This audit independently inspected both replay arms, but remains same-family and provisional.

Current V24/V27 language stays within this scope. The summary calls V24 non-confirmatory and not innovation efficacy (`results/contact_com_validation_v24_20261012/validation_summary.json:115-119`); V27 is diagnostic only and `innovation_confirmed=false` (`results/fixed_reference_com_v27_20261012/replay_summary.json:31-40`); README makes the same limitation explicit (`README.md:5`). Therefore the scope check passes. Any claim of comprehensive benchmark performance, robust cross-seed physics, semantic preservation, physical-quality success, candidate efficacy, or real-contact validity would exceed the evidence.

### F. Evaluation-type classification

- **Overall:** `simulation_only`.
- **Endpoint and heading conditions:** `real_gt_dataset_conditions_with_synthetic_offsets`; they are decoded from stored dataset motion but use a custom project split.
- **V24 continuation tracking:** `synthetic_proxy`; controller tracking against a frozen model-generated plan.
- **V27 raw replay:** `synthetic_proxy`; replay of the frozen generated reference.
- **V27 refined replay:** `synthetic_proxy`; replay of a frozen simulation-produced body trajectory.
- **Foot/contact measurement:** `simulation_only_calibrated_proxy`; not native ground-truth contact force and not arbitrary rolling-contact truth.
- **Semantics:** not verified by an independent evaluator (`scripts/physics_guided_metric_v23_20261012.py:158-160`).
- **Human evaluation / real robot:** absent.

## Fixed conditions, reset, clock, and same-configuration evidence

The collector binds the original target and case order, verifies replay original goals, uses an explicit frame clock, asserts the applied reference frame before physics, retains each active attempt until completion/termination, and saves an exact `0..steps-1` clock (`scripts/physics_guided_collect.py:121-139,190-211,226-241,274-280`). Reset alignment freezes the world frame after the actual reset and requires seven state comparisons plus four reset-cache comparisons to be zero before the first action (`scripts/reset_aligned_collect_v4.py:25-41,55-79`). Representative public metadata records all seven differences as zero and the reset gate as true (`results/fixed_reference_com_v27_20261012/public/raw/val_raw_replay_s11/metadata.json:18173-18183,18777-18795`).

All six V24 and two V27 saved `resolved_config.yaml` files are byte-identical, SHA-256 `b3eddac41eda4da9d0839c8d9de5aa98dbb608d9962a939958195a491c173218`; they contain `seed: 11` and `num_envs: 72`. V24 and V27 use the same frozen threshold SHA-256 `451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b`. V27 requests bind replay motion plus the current config, manifest, thresholds, and versions (`results/fixed_reference_com_v27_20261012/requests/val_raw_replay_s11.json:2-11`; corresponding refined request). The raw-byte config digest is `2d8fd0...`, and the frozen-threshold/source-manifest canonical parsed-JSON digest is `f7e06e...`; both independently recompute from the same file. Raw/refined critical metadata fields match the V24 base, including cases, names, `dt`, episode/context lengths, original/proposed goals and headings, foot geometry, native body properties, simulator parameters, seed, and round.

Read-only server inspection verified the omitted binary arrays without running simulation:

- both V27 trajectory archives contain finite 120-frame arrays for 72 environments; `valid` is `(120,72)`, `terminated` is `(72,)`, and `reference_frame` is exactly `0..119`;
- the raw and refined selected references are `(120,72,24,3)` float32 arrays reconstructed exactly from the V24 base reference or actual states, including terminal padding; two construction choices are incomplete, with valid lengths 95 and 109, and all 72 choices use candidate 0;
- raw replay equals the selected reference exactly; refined replay has an untransformed maximum difference `5.960464477539063e-08` m but equals the upstream float32 root-centering expression exactly. The pinned upstream implementation performs that subtract/re-add and coordinate transform at remote `third_party/CLoSD/closd/env/tasks/closd.py:651-704`, SHA-256 `dff1ca9bfb15c3058914c326cba83a07f69cd91952cd4e03ecfe1ad3d391e394`;
- independently recomputed request motion hashes match: raw `e570f591ddc30606c49156ba260e2e30d61cb5abf9fd5b50f3a607e9cb7dff3f`, refined `15602590194517214fef0ade9fd8f8a849fb98c07853d1015ce5112d56e5b838`;
- both replay arms record zero diffusion and generation-batch calls.

Remote binary SHA-256 values:

| Remote evidence path | SHA-256 |
|---|---|
| `results/contact_com_validation_v24_20261012/raw/val_planned_s11/trajectories.npz` | `45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd` |
| `results/fixed_reference_com_v27_20261012/selected/raw_reference.npz` | `80ba430d1cb6f108a860aacbcca6dd948b92ddeabce2d19f8cb5474858b468f2` |
| `results/fixed_reference_com_v27_20261012/selected/refined_reference.npz` | `7794442ba3cd0a65bd70da9f71c4f7ae883fb4b71cbe158ced309284d028a057` |
| `results/fixed_reference_com_v27_20261012/raw/val_raw_replay_s11/trajectories.npz` | `4f55270f61a6348ce803cc4b2632f04813e27d5c1317615e3d5bad20ec0cf054` |
| `results/fixed_reference_com_v27_20261012/raw/val_refined_replay_s11/trajectories.npz` | `960149fc8c5f07ce3a65573bc048aeb7044522857e2d2b46039280ddde12b0fa` |

## Claim impact

- **Supported:** V24 completed 432 retained validation attempts with 419 completions, 114 condition passes, 318 tracking passes, one physical-proxy pass, and one joint pass.
- **Supported, narrow:** V26 reproduces the cached first V24 batch under NumPy 1.23.5 with its enumerated preserved outputs unchanged and no resimulation.
- **Supported, needs qualifier:** V27 preserves the frozen target, reference clock, case order, thresholds, and construction-failure denominator for two 72-attempt replay arms. It is one generation seed and one shared construction batch.
- **Supported, proxy only:** the COM-corrected surface metric behaves consistently with the archived known-control calibration and fixed replay interface. It remains a body-force/collision-surface proxy.
- **Unsupported if asserted:** true contact force, arbitrary rolling-contact truth, verified semantic preservation, physical-quality success, candidate efficacy, official-test generalization, downstream utility, or real-robot validity.
- **Outside this audit:** V28 launch, queue progress, and any eventual formal-comparison result.

## Action items

1. Make the report gate check the exact V27 completion status, expected entries and counts, bound hashes, and diagnostic scope before rendering a pass state.
2. Apply the V24 new-batch reset/fence/sensor/contact guards to cached results as well, or bind an equivalent certificate that is verified on cache reuse.
3. Keep the digest representation explicit in future metadata: `2d8fd0...` is the raw-byte SHA-256 and `f7e06e...` is the sorted parsed-JSON SHA-256. No config mismatch was found.
4. Remove or explicitly mark unused V23 helpers and V24 imports, and document which surface fields are diagnostic versus gate inputs.
5. Preserve the current denominator, generation-seed wording, custom-validation-split disclosure, calibrated-proxy label, and no-efficacy/no-contact-truth limitations in every downstream claim.

## Audited primary-file hashes

| File | SHA-256 |
|---|---|
| `configs/physics_guided_20261011.json` | `2d8fd0e49ce7ed37393e1962dcd711b5a6e8f26de7e0ca4ef48964a6fd081502` |
| `scripts/physics_guided_metric_v23_20261012.py` | `b05b6ed8053ac037432471dabdf8608da383c76d4d2375ce10823e003f232c69` |
| `scripts/physics_guided_com_lineage_v24_20261012.py` | `ee440b92415ccc9646c29d1f300d2e0acd67daa9a369ce3c54816e01d4d2c850` |
| `scripts/contact_com_collect_v24_20261012.py` | `b3609d8a62aca242b77ec150262bac01d32b728c668b9a2a4912e468146aec11` |
| `scripts/run_contact_com_validation_v24_20261012.py` | `7e572184b0d341b607dbc67a624af17cf724e846e64045111804170c30e71d58` |
| `scripts/run_contact_com_runtime_v26_20261012.py` | `432051e1bb7510f8a37fb917f8da55348660843f002becb07885d79af6f863da` |
| `scripts/run_fixed_reference_com_v27_20261012.py` | `e1c83d0d621d48fad0e184fb6bec673c2474c96d7e4ddb547a3c8389c63d95e6` |
| `scripts/build_contact_measurement_report_20261011.py` | `a1de5cf5724ccff5d4296c108c1fbc0d5cbb5261b45764dbe77a43c3313017d2` |
| `results/contact_com_validation_v24_20261012/validation_summary.json` | `eab3c0c619748e0dcd7b1fde9de514aae868d973bcc8292576c8ad2187a49be8` |
| `results/contact_com_validation_v24_20261012/quality_diagnosis.json` | `9241401f2af15f3794597e63607c397a6a350149c44f6992e6523dd8e82e5c23` |
| `results/contact_com_runtime_v26_20261012/runtime_check.json` | `a17323fc1964255ec8f3644ae79a267b05b36c980a4e5d46d798d918c3f5129e` |
| `results/fixed_reference_com_v27_20261012/replay_summary.json` | `23971b708747128d476d0af7cb555dbd52cc3a7b901116817fb522c4c32db1e1` |
| `results/contact_measurement_review_20261011/measurement_review.html` | `6a914159ba970e0c2f5ff5d3e24f6526691ead510ce0baebd9561122fdc5208e` |
| `README.md` | `da1577e5d5c69dd4807ff6f69fc7a73e7ddd48f23beb76f053083bc73ddfd5be` |

This is a same-family review. Its acceptance is **provisional**, not cross-family acceptance.
