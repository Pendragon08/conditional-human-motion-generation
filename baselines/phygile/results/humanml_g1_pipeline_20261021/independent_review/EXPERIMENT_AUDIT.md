# Experiment Audit Report

**Date:** 2026-10-21  
**Auditor:** GPT-5.6-Sol ultra, fresh same-family reviewer  
**Review status:** same-family / provisional  
**Scope:** first SSM batch only  
**Overall verdict:** **WARN**  
**Integrity status:** **warn**

The first SSM batch passes the frozen document's deterministic data-preparation consistency command. Its raw-source provenance, official HumanML metadata, accepted manifests, stored hashes, recording-level split separation, and train-only normalization are internally consistent. The evidence ceiling is narrower than robot ground truth or model performance: the G1 targets are algorithmic retargeting proxies, thirteen source timebases remain physically ambiguous, several 262D and mirror semantics are not independently recomputed by the documented audit, and the effective test scope is two canonical recordings.

## Frozen document and document-following execution

The frozen document hash was identical immediately before and after execution: `52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093`. The exact output path was absent before the single invocation. The command at `docs/humanml_g1_processing_20261021.md:27-30` was:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/scripts/audit_humanml_g1_dataset.py --dataset /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_batch_v1 --public /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public --skeleton /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/configs/g1_codec_skeleton.json --gmr-root /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR --output /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json
```

- Command: 854 UTF-8 bytes without newline; SHA256 `195e69578199e504361b9903ffab9fa40f0f402be81d784d15651006fead9935`.
- UTC interval: `2026-10-21T13:19:05.1173119Z` to `2026-10-21T13:19:15.6717765Z`.
- Exit code: `0`.
- Stdout: 843 bytes; SHA256 `c280ae29ca419bd6b1c15a112ee78008695f6ec30465fe225c75a1574f55c6a8`.
- Stderr: 0 bytes; SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Receipt: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json`, 883 bytes, SHA256 `b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c`.

Exact stdout, followed by one LF:

```json
{"status": "DATASET_CONSISTENCY_CHECKS_PASSED", "evaluation_type": "data_preparation_consistency", "sequences_checked": 66, "frames_checked": 5904, "caption_lines_checked": 182, "raw_sources_rechecked": 30, "train_loader_sequences": 60, "position_max_abs_m": 3.6312626683354665e-08, "joint_max_abs_rad": 1.3802523524120147e-07, "rotation_matrix_max_abs": 1.2027506968159685e-07, "mujoco_vs_codec_fk_position_max_abs_m": 5.508160803024964e-07, "raw_source_and_derived_hashes_checked": true, "all_caption_frame_ranges_recomputed": true, "recording_disjoint_manifest_verified": true, "train_only_normalization_verified": true, "motion_tracking_success_evaluated": false, "author_exact_representation_verified": false, "generator_training_started": false, "audit_source_sha256": "c18a0412893bf392da77da5722825ac645623f4797846cb586d384d3328c63e4"}
```

The pre-freeze document was first read at SHA256 `e74f899662a03162c641a633b275c0553a4f5d12b6ed7ffc69fc7045d286163c` and changed after that invocation to `d42ce8a23b1d1607106fb44abd5f26bae80161e0445b2b2b4fca2f1b0e904ee2`. That first receipt remains a historical witness. The frozen `r2` execution closed this binding defect. The audit receipt itself still lacks the document hash, literal command, timestamps, exit code, and stream hashes, so it must remain paired with the external execution files in this directory, as required at `docs/humanml_g1_processing_20261021.md:35`.

No divergence occurred in the frozen `r2` execution. Historical divergences retained as evidence are: the first author invocation failed because `train.py` was absent (`results/humanml_g1_pipeline_20261021/audit_author_stdout.txt:1-6`); the required runtime sources were present for the final invocation; and the pre-freeze document changed after its successful command.

## A. Ground-truth and reference provenance — WARN

- HumanML metadata is pinned to commit `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23` (`docs/humanml_g1_processing_20261021.md:7`; `scripts/plan_humanml_amass.py:15`). Generation verifies the recorded hashes of `index.csv`, the text ZIP, and every split file (`scripts/process_humanml_g1.py:50-54`).
- The SSM SMPL+H archive was independently matched at 13,532,334 bytes and SHA256 `f3746e0d6f44354ae08d075208d47f5b5564934667429d56c6f2d21fcd376929` (`docs/humanml_g1_processing_20261021.md:9`). The audit recomputes all 30 raw hashes and marker timelines (`scripts/audit_humanml_g1_dataset.py:29-35`) and checks selected IDs against official split/source paths (`:40-49`). All 182 text lines matched the pinned official archive.
- Caption alignment follows the official integer floor stride and half-open crop convention (`scripts/plan_humanml_amass.py:36-51,54-83`).
- The source human motion and captions have real provenance. The G1 targets are generated by GMR and the independent codec and therefore are retargeting proxies, not measured robot ground truth. The codec states that it is not author-confirmed (`reimplementation/codec262.py:1-6`), and the batch records `author_data_or_performance_reproduced: false` (`results/humanml_g1_pipeline_20261021/ssm_batch_summary.json:45-49`).
- `scripts/process_humanml_g1.py:92-95` uses the undocumented height heuristic `1.66 + 0.1 * beta[0]`; the computed value is not bound into each source receipt. The documented audit checks stored native hashes but does not rerun raw SMPL-X through GMR, so it establishes integrity of recorded transformation artifacts rather than retargeting fidelity.

## B. Score normalization — PASS

No model performance score or prediction-derived denominator exists in this batch. Only channels `0:43` and `217:256` are standardized (`reimplementation/features.py:6-9`). Statistics are fitted from the supplied training rows (`reimplementation/features.py:29-39`), generation uses the source-disjoint manifest's train split (`scripts/process_humanml_g1.py:204-212`), and the audit reconstructs the means and standard deviations (`scripts/audit_humanml_g1_dataset.py:81-92`).

The actual manifest contains 60 train, 0 validation, and 6 test entries. Test subjects overlap train subjects, but test feature values do not enter normalization. The normalizer SHA256 is `b507e63a6981f15013f2b272a7bcf6a50ceeecbd6a234c209120ed9fd77c2793`. Sequence-row weighting repeats some canonical frames through nested caption segments and mirrors; this is a dataset-weighting property, not validation/test leakage.

## C. Result and artifact existence — WARN

- Current primary artifacts exist and agree: 30 requested/retargeted sources, 66 entries, 60 IDs, 5,904 sequence-frame visits, 182 captions, 33 mirrors, 0 rejects, and 60/0/6 train/validation/test entries (`remote summary.json:2-19`; `results/humanml_g1_pipeline_20261021/ssm_batch_summary.json:2-19`).
- The authoritative server inventory and `ssm_payload_manifest.json` match with zero missing, extra, size-mismatched, or hash-mismatched files: 266 files and 9,419,016 bytes (`results/humanml_g1_pipeline_20261021/ssm_payload_manifest.json:1335-1336`). The D-drive mirror also matches all 266 files (`D_drive_dataset_verification.json:2-7`).
- `source_timebase_summary.json:2-17` matches the 30 raw receipts. `rejected.json` is exactly `[]`. The frozen `r2` result reproduced the author result numerically.
- The boolean `raw_source_and_derived_hashes_checked` is broader than the checker: raw files are checked at `scripts/audit_humanml_g1_dataset.py:31-35`, and feature/native files at `:50-54`; metadata, retargeted-source files, manifests, normalization, rejected records, mirror witness, and public provenance are not all hashed by that command.
- The checker writes a receipt only on success (`scripts/audit_humanml_g1_dataset.py:20-21,104-106`). It has no atomic structured failure receipt. The first deployment failure survives only in separately preserved stdout.

## D. Dead code and check reachability — WARN

The document command reaches raw marker checks (`scripts/audit_humanml_g1_dataset.py:29-35`), official split/source checks (`:40-49`), feature/native shape/hash checks (`:50-57`), caption/raw-index recomputation (`:58-65`), codec decode comparisons (`:66-70`), sampled MuJoCo/FK comparisons (`:71-80`), recording overlap (`:81-85`), train normalization (`:86-92`), and `train.load_manifest` (`:93-94`).

Coverage gaps:

- The command does not call `mirror_fk_witness`, `mirror_native`, `quality`, or `codec.encode` (`reimplementation/humanml_pairing.py:71-141`; `scripts/process_humanml_g1.py:48,152-173`).
- Structural regressions in `scripts/test_humanml_pairing.py:22-54` are not run by the document command.
- `load_manifest` validates paths and split overlap (`reimplementation/train.py:23-65`), but the command does not instantiate `PairedMotionData`, its collation path, or a PyTorch `DataLoader` (`train.py:67-100`). The loader claim in `docs/humanml_g1_processing_20261021.md:33` and `reports/HumanML3D到G1配对数据处理报告.md:31` is too broad.
- The audit does not independently recompute every 262D channel. It discards the reported RIC redundancy residual (`reimplementation/codec262.py:203-208`; `scripts/audit_humanml_g1_dataset.py:67`), does not compare local body-velocity channels, and checks contacts only for binary range.
- MuJoCo/FK checks first/middle/last frames and body positions only (`scripts/audit_humanml_g1_dataset.py:71-78`). Manifest checking verifies source-group intersection but not full equality with `accepted.json` and official split labels. The current data passes independent direct comparison, but the command is underconstrained.

## E. Scope — WARN

The final document bounds the result to SSM, no validation set, no full HumanML3D/PhyGile result, and no training (`docs/humanml_g1_processing_20261021.md:37`; `.aris/compute/phygile-gmr-amass.md:29-35`). Effective scope is:

- 30 canonical recordings, 30 canonical IDs plus 30 `M` IDs, and 66 entries (33 unmirrored plus 33 mirrored).
- 5,904 sequence-frame visits, representing 2,852 unique canonical raw frames; nested captions repeat 100 canonical frame rows before mirroring.
- 182 official captions; 60/0/6 train/validation/test entries.
- The test bucket contains only two canonical recordings, two canonical IDs, and 87 unique canonical raw frames.
- Three subject/session identifiers occur; both test identifiers also occur in train. Recording intersections are empty, but the split is not subject-disjoint.

Timebase boundary: caption bounds use `int(seconds * 20)` and half-open slices (`scripts/plan_humanml_amass.py:36-51,76-80`); source stride is `int(reference_fps / 20)` (`:67-75`; `scripts/process_humanml_g1.py:81-85`). Thirteen sources have reference headers near 60 Hz while SMPL-X reports 120 Hz; seventeen have both near 120 Hz (`source_timebase_summary.json:3-13`). Marker frames are identical despite these header conflicts. The selected stride follows the nominal HumanML convention, but physical 20 Hz is not established for the thirteen ambiguous sources. Velocity and speed-threshold units labeled m/s or rad/s (`reimplementation/humanml_pairing.py:54-60,123-135`) require that qualifier.

Missing markers: only NaN triplets or all-zero XYZ triplets are treated as absent, and normalized masks must match (`reimplementation/humanml_pairing.py:20-39`). Every receipt passed; maximum measured-marker error was `0.0`, and the minimum visible common-marker count was 44 (`source_timebase_summary.json:15-16`). `ssm_marker_visibility_probe.json:8-14,36-42,64-70` shows raw finite masks alone would be insufficient.

Mirroring is anatomical rather than metadata-only: root pose is reflected, left/right joint identities are swapped, joint signs come from URDF axes, and velocities are recomputed (`reimplementation/humanml_pairing.py:71-85`; `scripts/process_humanml_g1.py:136-167`). Official `M` text is loaded directly. The stored 16-pose witness reports about `9.711e-06 m` body-position residual, `5.55e-16` rotation residual, and zero double-mirror joint residual under a `2e-5 m` tolerance. The document audit does not recompute every stored base/`M` pair, so artifact-level mirroring retains a coverage qualifier.

FK/skeleton: 29 named joints and provenance are recorded (`configs/g1_codec_skeleton.json:1083-1116`), and GMR/MuJoCo addresses are resolved by joint name (`scripts/process_humanml_g1.py:96-101`; `scripts/audit_humanml_g1_dataset.py:36-38`). The sampled MuJoCo-versus-codec maximum position error is `5.508160803024964e-07 m`. This supports limited kinematic convention agreement, not dynamics, collision, torque, tracking, or all-frame/all-body FK identity.

## F. Evaluation classification — WARN

**Classification:** `other:mixed_real_source_algorithmic_retarget_proxy`  
**Audit type:** `data_preparation_consistency`

Real-source components are licensed SSM motion exports, measured marker trajectories, official HumanML split metadata, and official text. G1 native motion and 262D targets are generated algorithmically by GMR and an independent codec. This is not real robot ground truth, model performance evaluation, human evaluation, physical tracking, or simulation performance.

The supported claim ceiling is provenance, current-file integrity, exact caption/index alignment under the selected nominal HumanML convention, recording-level split isolation, train-only normalization, and limited codec/FK consistency. Unsupported above that ceiling are physical 20 Hz for all sources, subject generalization, author-exact 262D equivalence, full GMR fidelity, dynamics/collision/torque feasibility, tracking success, generator quality, paper metrics, complete HumanML3D coverage, or PhyGile reproduction.

## Must-fix items

1. Resolve the thirteen-source physical-timebase ambiguity or consistently label alignment as nominal HumanML indexing; do not claim physical m/s or rad/s for those sources without an authoritative timebase.
2. Compare every manifest entry and bucket with `accepted.json` and the official split map before fitting normalization. Preserve the recording-level qualifier and prohibit subject-disjoint claims for this batch.
3. Independently re-encode and compare all 262 channels, including RIC, local body velocities, and contact semantics; threshold the existing RIC residual.
4. Pair every base/`M` artifact and recompute anatomical reflection across native, FK, feature, and official caption representations.
5. Receipt the per-source height value, label `1.66 + 0.1 * beta[0]` as a heuristic, and bind the full GMR solver source/version before claiming retargeting fidelity.
6. Correct loader wording: the command calls manifest loading and input validation, not `PairedMotionData` or a PyTorch `DataLoader`.
7. Atomically preserve success and failure records with document/command/source/data hashes, timestamps, exit code, and stream hashes. The external frozen `r2` trace closes this run's binding defect; the script still lacks a failure receipt.
8. Report effective scope with every downstream result: 30 recordings, three subjects, no validation, two test recordings, 87 unique canonical test frames, subject overlap, 2,852 unique canonical frames, and 5,904 mirrored/repeated sequence-frame visits.
9. Correct `reports/HumanML3D到G1配对数据处理报告.md`: line 35 should use the recorded `3.63e-8` and `1.38e-7` residuals; line 37 must cite the companion command/stdout/stderr/receipt files and avoid a dangling audit-report reference; lines 12/28 must state the official `M` ID and algorithmic proxy boundary; lines 17/21/31 must disclose effective test scope; and the `8 + 6 tests passed` claim at line 37 needs a preserved test log or implementer-reported qualifier.

## Claim impact

| Claim | Impact | Boundary |
|---|---|---|
| Official SSM/HumanML source and caption provenance | supported | Pinned inputs, paths, hashes, and captions verified |
| 30 sources, 66 entries, 5,904 frames, 182 captions | needs qualifier | 5,904 is sequence-frame visits with mirrors and repeated caption segments |
| Physical 20 Hz for all sources | unsupported | Thirteen sources have conflicting 60/120-Hz headers |
| Recording-disjoint split | supported | Actual recording intersections are empty |
| Subject-disjoint split | unsupported | Both test subject/session IDs occur in train |
| Train-only normalization | supported | Recomputed from the 60-entry train bucket |
| Anatomical mirroring of stored artifacts | needs qualifier | Code and witness support it; every stored pair is not independently recomputed |
| Full 262-channel semantic equivalence | needs qualifier | Root/joint/rotation paths pass; other channel semantics are incomplete |
| Real robot ground truth, tracking, dynamics, or generator performance | unsupported | Proxy targets; tracking and training did not run |
| Complete HumanML3D or PhyGile reproduction | unsupported | First SSM batch only |
| Frozen final-document command binding | supported | Pre/post document hashes match and exact external trace is retained |

## Review limitation

This is a fresh same-family Codex review, so positive findings are provisional. Deterministic hashes and the frozen `r2` execution evidence are directly reproducible; the substantive integrity verdict is not cross-family acceptance.
