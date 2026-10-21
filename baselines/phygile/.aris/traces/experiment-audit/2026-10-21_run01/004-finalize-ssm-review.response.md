# Fresh experiment-integrity review

**Overall verdict: WARN — same-family, provisional.**

The frozen `r2` command passed exactly as documented and closes the earlier document-binding defect for this final document version. The first SSM batch is internally consistent with its recorded raw sources, official HumanML metadata, accepted manifests, train-only normalization, and stored hashes.

The evidence supports a bounded **data-preparation consistency** result. It does not establish robot-motion ground truth, physical timing for all recordings, GMR fidelity, full 262-channel semantic correctness, dynamics, tracking, generator training quality, or PhyGile performance.

## Frozen-document execution

The final document on disk matched its declared frozen hash:

- Document: `docs/humanml_g1_processing_20261021.md`
- SHA256 before: `52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093`
- SHA256 after: `52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093`
- The output path was confirmed absent before execution.
- Exact command, from `docs/humanml_g1_processing_20261021.md:27-30`:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/scripts/audit_humanml_g1_dataset.py --dataset /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_batch_v1 --public /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public --skeleton /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/configs/g1_codec_skeleton.json --gmr-root /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR --output /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json
```

- Command UTF-8 length: 854 bytes
- Command SHA256, no newline: `195e69578199e504361b9903ffab9fa40f0f402be81d784d15651006fead9935`
- Started UTC: `2026-10-21T13:19:05.1173119Z`
- Ended UTC: `2026-10-21T13:19:15.6717765Z`
- Exit code: `0`
- Stdout: 843 bytes, SHA256 `c280ae29ca419bd6b1c15a112ee78008695f6ec30465fe225c75a1574f55c6a8`
- Stderr: empty, 0 bytes, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Exact stdout, followed by one LF:

```json
{"status": "DATASET_CONSISTENCY_CHECKS_PASSED", "evaluation_type": "data_preparation_consistency", "sequences_checked": 66, "frames_checked": 5904, "caption_lines_checked": 182, "raw_sources_rechecked": 30, "train_loader_sequences": 60, "position_max_abs_m": 3.6312626683354665e-08, "joint_max_abs_rad": 1.3802523524120147e-07, "rotation_matrix_max_abs": 1.2027506968159685e-07, "mujoco_vs_codec_fk_position_max_abs_m": 5.508160803024964e-07, "raw_source_and_derived_hashes_checked": true, "all_caption_frame_ranges_recomputed": true, "recording_disjoint_manifest_verified": true, "train_only_normalization_verified": true, "motion_tracking_success_evaluated": false, "author_exact_representation_verified": false, "generator_training_started": false, "audit_source_sha256": "c18a0412893bf392da77da5722825ac645623f4797846cb586d384d3328c63e4"}
```

Receipt:

- Path: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json`
- Size: 883 bytes
- Mtime: `2026-10-21 13:19:17.421973409 +0000`
- SHA256: `b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c`
- Content: the same fields and values as stdout, pretty-printed, ending with LF.

The historical first document was read at SHA256 `e74f899662a03162c641a633b275c0553a4f5d12b6ed7ffc69fc7045d286163c`; after its invocation it changed to `d42ce8a23b1d1607106fb44abd5f26bae80161e0445b2b2b4fca2f1b0e904ee2`. Therefore `ssm_dataset_audit_document.json` is only a historical witness. The final document explicitly records that limitation at `docs/humanml_g1_processing_20261021.md:35`. The frozen `r2` pre/post hashes are identical, so the final command-to-document binding now passes.

The receipt itself still contains no document hash, literal command, timestamps, exit code, or stream hashes. The separate execution trace required by `docs/humanml_g1_processing_20261021.md:35` must be retained with it. The author receipt, historical receipt, and `r2` receipt all have identical content hash `b60eca...`; the external trace is what distinguishes the executions.

## A. Ground-truth and reference provenance — **WARN**

Supported evidence:

- HumanML metadata is pinned to commit `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`: `docs/humanml_g1_processing_20261021.md:7`, `scripts/plan_humanml_amass.py:15`.
- Generation checks the recorded hashes of `index.csv`, text ZIP, and all split files before processing: `scripts/process_humanml_g1.py:50-54`.
- The public provenance record contains source URLs and hashes at remote `code/public/provenance.json:2-5,20-53`, with counts at `:62-77`. I independently matched the six files used here and verified the text ZIP CRC.
- The SSM SMPL+H archive was independently verified at 13,532,334 bytes and SHA256 `f3746e0d6f44354ae08d075208d47f5b5564934667429d56c6f2d21fcd376929`, as claimed at `docs/humanml_g1_processing_20261021.md:9`.
- The audit recomputes all 30 raw source hashes and marker timelines: `scripts/audit_humanml_g1_dataset.py:29-35`.
- Every selected motion ID is checked against the official split and source path: `scripts/audit_humanml_g1_dataset.py:40-49`.
- Caption time handling matches the pinned official floor-stride and half-open slicing convention: `scripts/plan_humanml_amass.py:36-51,54-83`. This is consistent with the pinned HumanML3D processing notebook.
- All 182 selected text lines were independently matched to their official archive members.

Limitation:

- The human source motion and captions have real-source provenance. The G1 “target” is produced algorithmically by GMR and is therefore a retargeting proxy, not measured robot ground truth.
- The codec explicitly says it is independent and not author-confirmed: `reimplementation/codec262.py:1-6`; the batch records `author_data_or_performance_reproduced: false` at `results/humanml_g1_pipeline_20261021/ssm_batch_summary.json:45-49`.
- `scripts/process_humanml_g1.py:92-95` passes `1.66 + 0.1 * beta[0]` as `actual_human_height`. This undocumented height heuristic affects GMR scaling and is not stored as a per-source receipt field.
- The independent command does not rerun raw SMPL-X → GMR retargeting. It verifies raw hashes and previously stored native hashes, so it establishes integrity of the recorded transformation artifacts, not retargeting fidelity.

No artifact presented an output-derived robot target as real ground truth, but downstream descriptions must retain the proxy label.

## B. Score normalization — **PASS**

There are no performance scores or self-derived evaluation denominators in this batch.

- The 262D layout is explicit at `reimplementation/features.py:6-9`. Only channels `0:43` and `217:256` are standardized; rotations and binary contacts retain their native conventions.
- Normalizer fitting concatenates only the supplied training sequences: `reimplementation/features.py:29-39`.
- Generation fits `normalization_train_only.npz` from `manifest_source_disjoint.json`’s train split: `scripts/process_humanml_g1.py:204-212`.
- The audit independently reconstructs those means and standard deviations and compares them to the NPZ: `scripts/audit_humanml_g1_dataset.py:81-92`.
- I additionally verified that the actual source-disjoint manifest equals the accepted entries with the correct official split labels. It contains 60 train, 0 validation, and 6 test sequence entries.
- `normalization_train_only.npz` SHA256: `b507e63a6981f15013f2b272a7bcf6a50ceeecbd6a234c209120ed9fd77c2793`.

The statistics are sequence-row weighted, so mirrored rows and nested caption segments can repeat canonical source frames. This is an explicit dataset-weighting property, not validation/test value leakage. Test subjects overlap train subjects, but no test feature rows enter the normalizer.

## C. Result and artifact existence/consistency — **WARN**

All presently claimed first-batch artifacts exist and their current bytes are internally consistent:

- Remote `summary.json:2-19` records 30 requested/retargeted sources, 66 sequences, 60 IDs, 5,904 frame rows, 182 captions, 33 mirrored entries, 0 rejects, 60/0/6 split counts.
- The same values appear at `results/humanml_g1_pipeline_20261021/ssm_batch_summary.json:2-19`, and all shared fields match the authoritative remote summary.
- The summary’s environment and scope limitations are at `ssm_batch_summary.json:21-49`.
- Remote inventory contains exactly 66 feature arrays, 66 native arrays, 66 metadata files, 30 retargeted-source arrays, 30 source receipts, 8 other top-level files, 266 total files, and 9,419,016 bytes.
- I compared every remote path, byte length, and SHA256 against `ssm_payload_manifest.json`; there were zero missing, extra, or mismatched files. Totals are recorded at `ssm_payload_manifest.json:1335-1336`.
- I separately compared the D-drive mirror and found the same 266 files, 9,419,016 bytes, with zero mismatches or extras. Its recorded claim is at `D_drive_dataset_verification.json:2-7`.
- `source_timebase_summary.json:2-17` matches all 30 raw source receipts exactly.
- `rejected.json` is exactly `[]`.
- The final `r2` receipt reproduced the author’s numeric result exactly.

Primary hashes:

| Artifact | SHA256 |
|---|---|
| Remote summary | `d1c5f4859255989c71784df4bd54eddf2abecd6d0ed41eaebcf5de5e5a918e9a` |
| Accepted entries | `ed0e76f98646bdd15d0934041d486bb57183cc59bb6ebbcfc7401d4f7b96793a` |
| Rejected entries | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| Official manifest | `2df00eab7f9a26f0e8494bf412f679886eaf831bbf2d887c8a4ead86dfc318c1` |
| Source-disjoint manifest | `35aa58fe79b726322a7c3a12bf562cd820e35693bbdf0de389078753cdaad193` |
| Train-only normalization | `b507e63a6981f15013f2b272a7bcf6a50ceeecbd6a234c209120ed9fd77c2793` |
| Mirror witness | `0ce63509e18205a3f337f9534b5112737e5b0fca2880b50a62904c6319c08c89` |
| Pairing config | `7256ca9a084e783d83ff6765df4e22ad7a9b90f97dede47a00ec8f5b91691cc9` |
| Skeleton config | `d2627ab9be14d9dd46fc483b754128e6c8f2515db5b1a274ba1b2b97d9aac6e7` |
| Payload manifest | `15ff4c1ccc2285b6618f3845fddadba21fc39b9c105ce506748d80c10a6b10ad` |
| D-drive verification | `f5506e76e115291a4df40bc68060ca2a1f74b69552ead6e2a5d2539222bcce71` |
| Timebase summary | `81e7353a36c93fbbe5414aa5f35c850b4b2f44b61240c6a707a100728c022662` |
| Local batch summary | `0c771ee52c88e501b4b20cac0a39ec10e68bb1aede809614e9d0098ebe0429db` |

Warnings:

- The audit result’s `raw_source_and_derived_hashes_checked: true` is broader than the implementation. Lines `31-35` check raw files; lines `50-54` check feature/native files. The audit does not hash codec metadata, retargeted-source files, manifests, normalization, rejected records, mirror witness, or public provenance.
- The author’s first run failed because `train.py` was absent: `results/humanml_g1_pipeline_20261021/audit_author_stdout.txt:1-6`. Its later successful stdout is in `audit_author_r2_stdout.txt:1`.
- Remote `train.py`, `model.py`, `text.py`, and `diffusion.py` are now present and match their local byte hashes. All sources needed by the final command were present before `r2`.
- `scripts/audit_humanml_g1_dataset.py:20-21,104-106` writes a receipt only after success. It has no immutable structured failure receipt. The first failure survived only because stdout was separately retained.
- Generation records per-source exceptions at `scripts/process_humanml_g1.py:190-193`, but an interruption or post-loop failure has no complete run-level failure record.

## D. Dead code and check reachability — **WARN**

Reached by the documented command:

- Raw marker verification: `audit_humanml_g1_dataset.py:29-35`
- Official split/source checks: `:40-49`
- Feature/native hashes and basic shape/range checks: `:50-57`
- Caption and raw-frame recomputation: `:58-65`
- Codec decode/root/joint comparison: `:66-70`
- MuJoCo/FK body-position comparison: `:71-80`
- Recording-group overlap check: `:81-85`
- Train-only normalization recomputation: `:86-92`
- `train.load_manifest`: `:93-94`

Not reached or incompletely reached:

- The documented audit does not call `mirror_fk_witness`, `mirror_native`, `quality`, or `codec.encode`. Those generation functions are at `humanml_pairing.py:71-141` and `process_humanml_g1.py:48,152-173`.
- `scripts/test_humanml_pairing.py:22-54` contains useful synthetic regressions, but the documented command does not execute them.
- `load_manifest` validates files and split overlap at `reimplementation/train.py:23-65`, but the command does not instantiate `PairedMotionData`, its collation path, or a PyTorch `DataLoader` at `train.py:67-100`. Therefore “实际调用生成器的数据加载器” in `docs/humanml_g1_processing_20261021.md:33` overstates what ran.
- The audit checks the reconstructed root position, root rotation, and 29 joint angles. It does not independently recompute all 262 channels: `decode` reports RIC redundancy at `codec262.py:203-208`, but the audit discards the report at `audit_humanml_g1_dataset.py:67`; local-body velocity channels are not compared; contact channels are checked only for binary values.
- MuJoCo FK is sampled at first/middle/last frame only and checks positions only: `audit_humanml_g1_dataset.py:71-78`. It does not check all frames, body rotations, or the fixed rubber-hand bodies used by selected feature blocks.
- The manifest audit verifies only source-group intersection. It does not prove manifest membership equals `accepted.json` or that each manifest bucket agrees with `entry.official_split`. The present manifest passes an independent direct comparison, but the documented checker would not detect a malicious split relabel.
- The independent command does not recompute raw motion through GMR, public provenance hashes, mirror witness, filtering, rejected records, or the payload manifest.

## E. Scope — **WARN**

The final document correctly limits the claim to SSM, no validation split, no full HumanML3D/PhyGile result, and no training: `docs/humanml_g1_processing_20261021.md:37`. The environment ledger makes the same boundary at `.aris/compute/phygile-gmr-amass.md:29-35`.

Exact first-batch scope:

- 30 canonical SSM recordings
- 30 canonical motion IDs plus 30 `M` IDs
- 66 sequence entries: 33 unmirrored plus 33 mirrored
- 5,904 sequence-frame visits: 2,952 unmirrored plus 2,952 mirrored
- 2,852 unique canonical raw source frames
- 100 repeated canonical frame rows from nested caption segments, before mirroring
- 182 official caption lines
- Train: 60 entries, derived from 28 canonical recordings
- Validation: 0
- Test: 6 entries, but only 2 canonical recordings, 2 canonical motion IDs, and 87 unique canonical raw frames
- Three subject/session identifiers total
- Both test subject/session identifiers also occur in train

Consequently:

- `5,904 frames` must be described as sequence-frame rows or visits, not independent measurements.
- The manifest is recording-disjoint, and actual recording intersections are empty.
- It is not subject-disjoint. A subject-generalization claim is unsupported.
- The test set is too small for broad model-performance conclusions.
- Seed `42` is fixed in `configs/humanml_pairing.json:3-5`; it governs the processing setup and 16-pose mirror witness. No generator training seed or trained model exists.

### Time and caption alignment

- Caption boundaries use `int(seconds * 20)` and half-open slicing: `scripts/plan_humanml_amass.py:36-51,76-80`.
- Source downsampling uses `int(reference_fps / 20)`: `plan_humanml_amass.py:67-75`, `process_humanml_g1.py:81-85`.
- The audit recomputes each accepted caption’s exact raw indices: `audit_humanml_g1_dataset.py:58-65`.
- Thirteen sources have reference headers `59.9999885559082` versus SMPL-X `120.0`; seventeen have reference headers `120.00004577636719` versus SMPL-X `120.0`: `source_timebase_summary.json:3-13`.
- Markers are frame-identical despite the conflicting headers. The selected stride of 2 for the thirteen sources follows the official HumanML convention, but it does not establish physical 20 Hz. If the reference header were literal, the sampled physical rate would be about 30 Hz. Because the two headers conflict, neither physical interpretation is established.
- Derivatives are nevertheless calculated with `fps=20`: `humanml_pairing.py:54-60`; speed thresholds are labeled m/s and rad/s at `:123-135`. Those physical-unit claims need qualification for the thirteen ambiguous sources.

### Missing markers

- The implementation treats only NaN triplets or all-zero XYZ triplets as absent and requires identical normalized masks: `humanml_pairing.py:20-39`.
- It requires at least 30 common markers, over 50% valid observations, at least 10 valid markers per frame, and at most `1e-7 m` mismatch.
- All 30 receipts passed with measured-marker maximum error `0.0`; the observed minimum visible common-marker count was 44: `source_timebase_summary.json:15-16`.
- `ssm_marker_visibility_probe.json:8-14,36-42,64-70` shows why raw finite masks alone are insufficient: some zero-versus-NaN encodings differ while normalized missing masks agree.

### Mirroring

- This is an anatomical robot-space transformation, not a metadata-only relabel or simple coordinate sign flip.
- `mirror_native` reflects root pose, swaps left/right joint identities, derives joint signs from URDF axes, and recomputes velocities: `humanml_pairing.py:71-85`.
- The process applies it to each `M` motion before feature encoding: `process_humanml_g1.py:136-167`.
- Official `M` text is loaded directly, so the code does not perform a second text substitution: `process_humanml_g1.py:136-146`.
- I compared the selected base/`M` captions; directional wording changes present in official `M` text were retained.
- The stored seed-42, 16-pose witness reports body-position residual about `9.711e-06 m`, rotation residual about `5.55e-16`, and zero double-mirror joint residual, within its `2e-5 m` tolerance.
- The documented audit does not independently relate every stored mirrored feature/native artifact to its unmirrored partner. Therefore artifact-level anatomical mirroring remains supported by generation code and the stored witness, with an audit-coverage qualifier.

### FK and skeleton conventions

- The skeleton records 29 named joints and its schema/provenance at `configs/g1_codec_skeleton.json:1083-1116`.
- GMR qpos addresses are resolved explicitly by joint name: `process_humanml_g1.py:96-101`.
- Audit-side MuJoCo addresses are independently resolved by the same names: `audit_humanml_g1_dataset.py:36-38`.
- Current URDF, GMR XML, and IK-config hashes match their source receipts.
- The recorded MuJoCo-versus-codec maximum sampled body-position difference is `5.508160803024964e-07 m`.
- This establishes sampled kinematic convention agreement, not dynamics, collision behavior, tracking success, or comprehensive all-body/all-frame FK identity.

## F. Evaluation classification — **WARN**

**Precise classification:** `other:mixed_real_source_algorithmic_retarget_proxy`

- Real-source components: licensed SSM motion exports, measured marker trajectories, official HumanML split metadata, and official text.
- Proxy component: G1 native motion and 262D targets generated algorithmically by GMR and the independent codec.
- Evaluation type: `data_preparation_consistency`.
- It is not `real_gt` for robot motion.
- It is not a model-performance evaluation.
- It is not human evaluation.
- It is not a simulation or physical tracking result.
- It is not self-supervised metric normalization, although some checks are encode/decode self-consistency checks.

**Claim ceiling:** provenance, file integrity, exact caption/index alignment under the selected nominal HumanML convention, recording-level split isolation, stored-feature train-only normalization, and limited codec/FK consistency.

Unsupported above that ceiling: physical 20 Hz for all sources, subject-generalization, author-exact 262D equivalence, full GMR fidelity, dynamic feasibility, mesh collision safety, torque feasibility, GMT/GMR tracking success, trained generator quality, paper metrics, complete HumanML3D coverage, or PhyGile reproduction.

## Must-fix issues

1. **Resolve or explicitly encode the thirteen-source physical-timebase ambiguity.** Retain “nominal 20 Hz HumanML indexing” for alignment, but do not describe velocities or threshold units as physical m/s or rad/s until the conflicting headers have an authoritative resolution.
2. **Strengthen manifest verification.** The audit must compare every manifest entry and split bucket against `accepted.json` and the official split map before fitting normalization. Recording disjointness must remain explicitly recording-level; subject-disjoint claims must be prohibited for this batch.
3. **Audit all 262 channels semantically.** Re-encode native arrays and compare root channels, RIC positions, rotations, local body velocities, and contacts. Threshold the existing RIC residual instead of discarding it.
4. **Add artifact-level mirror verification.** Pair every base/`M` native artifact, recompute anatomical reflection, and compare exact root, joints, FK, feature blocks, and official `M` captions.
5. **Record retarget-scale provenance.** Store the computed human-height value per source, label the `1.66 + 0.1 * beta[0]` formula as a heuristic, and bind the complete GMR solver source/version rather than only IK/XML hashes. Recompute GMR if a retarget-fidelity claim is required.
6. **Correct the loader claim.** The documented command calls `load_manifest`; it does not instantiate `PairedMotionData` or a PyTorch `DataLoader`.
7. **Preserve immutable failure evidence.** Future audits should atomically create a run record for both success and failure, including document hash, literal command hash, timestamps, exit code, stdout/stderr hashes, audit-source hash, and primary dataset hashes. The frozen `r2` external trace closes the current document-binding defect, but the audit script itself still has no failure receipt.
8. **Report effective sample scope with every downstream result.** Include 30 recordings, three subjects, no validation set, two test recordings, subject overlap, 2,852 unique canonical frames, and the fact that 5,904 counts mirrored and repeated sequence-frame visits.

## Claim impacts

| Claim | Impact | Finding |
|---|---|---|
| Official HumanML/SSM input provenance | `supported` | Pinned files, archive, source paths, raw hashes, and captions verified |
| Thirty raw sources have frame-identical measured markers | `supported` | All receipts recomputed; max observed marker error 0 |
| 66 entries / 5,904 frames / 182 captions | `needs_qualifier` | Counts are correct; 5,904 is sequence-frame visits, not unique measurements |
| Canonical physical 20 Hz | `unsupported` for 13 sources | Only the nominal HumanML indexing convention is established |
| Recording-disjoint manifest | `supported` | Actual recording intersections are empty |
| Subject-disjoint evaluation | `unsupported` | Both test subjects also occur in train |
| Train-only normalization | `supported` | Independently recomputed from the correct 60-entry train bucket |
| Anatomical mirroring | `needs_qualifier` | Implementation and witness support it; documented audit does not recompute every stored pair |
| Full 262-channel semantic correctness | `needs_qualifier` | Root/joint/rotation paths pass; RIC, local velocities, and contact semantics are not independently recomputed |
| Real robot ground truth | `unsupported` | G1 targets are algorithmic retargeting proxies |
| Dynamics, collision, torque, or tracking success | `unsupported` | Explicitly not evaluated |
| Generator/model performance | `unsupported` | Training has not started |
| Complete HumanML3D or PhyGile reproduction | `unsupported` | First SSM batch only |
| Frozen final-document audit binding | `supported` | Final pre/post document hash identical; exact external execution trace captured |

## JSON-ready record

```json
{
  "overall_verdict": "WARN",
  "reviewer_independence": "fresh_same_family",
  "provisional": true,
  "scope": "first_ssm_batch_only",
  "evaluation_classification": {
    "type": "other:mixed_real_source_algorithmic_retarget_proxy",
    "audit_type": "data_preparation_consistency",
    "claim_ceiling": "provenance, integrity, nominal HumanML alignment, recording-level split isolation, train-only normalization, and limited codec/FK consistency"
  },
  "reason_codes": [
    "PROXY_TARGET_CLASSIFICATION_REQUIRED",
    "PHYSICAL_TIMEBASE_AMBIGUOUS_13_OF_30",
    "FULL_262_SEMANTIC_AUDIT_INCOMPLETE",
    "MIRROR_ARTIFACT_AUDIT_INCOMPLETE",
    "MANIFEST_CHECKER_UNDERCONSTRAINED",
    "SUBJECT_OVERLAP_ACROSS_TRAIN_TEST",
    "TEST_SCOPE_TWO_CANONICAL_RECORDINGS",
    "RETARGET_HEIGHT_HEURISTIC_NOT_RECEIPTED",
    "LOADER_CLAIM_OVERSTATED",
    "IMMUTABLE_FAILURE_RECEIPT_MISSING",
    "HISTORICAL_DOCUMENT_MUTATION_CLOSED_BY_FROZEN_R2"
  ],
  "frozen_document": {
    "sha256_expected": "52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093",
    "sha256_before": "52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093",
    "sha256_after": "52bbcc956ee87eeccbea9abf1b7b48dfa1cf6230bdffe50d7a74cace4c5db093",
    "unchanged_during_execution": true
  },
  "audit_execution": {
    "output_absent_before": true,
    "command_sha256_utf8_no_newline": "195e69578199e504361b9903ffab9fa40f0f402be81d784d15651006fead9935",
    "started_utc": "2026-10-21T13:19:05.1173119Z",
    "ended_utc": "2026-10-21T13:19:15.6717765Z",
    "exit_code": 0,
    "stdout_bytes": 843,
    "stdout_sha256": "c280ae29ca419bd6b1c15a112ee78008695f6ec30465fe225c75a1574f55c6a8",
    "stderr_bytes": 0,
    "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "receipt_path": "/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json",
    "receipt_bytes": 883,
    "receipt_sha256": "b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c"
  },
  "statuses": {
    "A_ground_truth_reference_provenance": "WARN",
    "B_score_normalization": "PASS",
    "C_result_artifact_consistency": "WARN",
    "D_dead_code_reachability": "WARN",
    "E_scope": "WARN",
    "F_evaluation_classification": "WARN"
  },
  "counts": {
    "canonical_recordings": 30,
    "canonical_motion_ids": 30,
    "motion_ids_including_mirror": 60,
    "sequence_entries": 66,
    "sequence_frame_visits": 5904,
    "unique_canonical_raw_frames": 2852,
    "caption_lines": 182,
    "train_entries": 60,
    "validation_entries": 0,
    "test_entries": 6,
    "test_canonical_recordings": 2,
    "subjects_or_sessions": 3,
    "test_subjects_overlapping_train": 2
  },
  "hashes": {
    "environment_ledger": "4dfcb5d66afd5c9ce353aedf37027685d8a410b17e840b534bb242e06c418f96",
    "audit_source": "c18a0412893bf392da77da5722825ac645623f4797846cb586d384d3328c63e4",
    "process_source": "73fa180339c3fd3f43ee13f7e6708170b183a793d52fe1894f078c13af89da6e",
    "pairing_source": "4f6db72b641665eb82425d13a0e0e0b28339fc0bf2fcf6eab3edcb9706ccb158",
    "codec_source": "e95cc8f8f782cc0577819e862328e814a4d8e8b17bea5bee0efa46b6a194606e",
    "planning_source": "2e08917e2dead829efa8632a4b475067c5054f2e6277b3fe6cbc5bae7b131369",
    "pairing_config": "7256ca9a084e783d83ff6765df4e22ad7a9b90f97dede47a00ec8f5b91691cc9",
    "skeleton_config": "d2627ab9be14d9dd46fc483b754128e6c8f2515db5b1a274ba1b2b97d9aac6e7",
    "source_archive": "f3746e0d6f44354ae08d075208d47f5b5564934667429d56c6f2d21fcd376929",
    "public_provenance": "0ccba68da6bb7c663d2f2d73581fc3bc07a028968b71ed075125a8948ab49c8e",
    "dataset_summary": "d1c5f4859255989c71784df4bd54eddf2abecd6d0ed41eaebcf5de5e5a918e9a",
    "accepted": "ed0e76f98646bdd15d0934041d486bb57183cc59bb6ebbcfc7401d4f7b96793a",
    "rejected": "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
    "manifest_official": "2df00eab7f9a26f0e8494bf412f679886eaf831bbf2d887c8a4ead86dfc318c1",
    "manifest_source_disjoint": "35aa58fe79b726322a7c3a12bf562cd820e35693bbdf0de389078753cdaad193",
    "normalization_train_only": "b507e63a6981f15013f2b272a7bcf6a50ceeecbd6a234c209120ed9fd77c2793",
    "payload_manifest": "15ff4c1ccc2285b6618f3845fddadba21fc39b9c105ce506748d80c10a6b10ad"
  },
  "claims": [
    {"claim": "official_source_and_caption_provenance", "impact": "supported"},
    {"claim": "counts_66_5904_182_30", "impact": "needs_qualifier", "qualifier": "5904 is sequence-frame visits including mirrors and repeated caption segments"},
    {"claim": "physical_20hz_for_all_sources", "impact": "unsupported"},
    {"claim": "recording_disjoint_split", "impact": "supported"},
    {"claim": "subject_disjoint_split", "impact": "unsupported"},
    {"claim": "train_only_normalization", "impact": "supported"},
    {"claim": "anatomical_mirroring_of_stored_artifacts", "impact": "needs_qualifier"},
    {"claim": "full_262_channel_semantic_equivalence", "impact": "needs_qualifier"},
    {"claim": "real_robot_ground_truth_or_tracking_performance", "impact": "unsupported"},
    {"claim": "generator_training_or_model_performance", "impact": "unsupported"},
    {"claim": "final_frozen_document_execution_binding", "impact": "supported"}
  ]
}
```
