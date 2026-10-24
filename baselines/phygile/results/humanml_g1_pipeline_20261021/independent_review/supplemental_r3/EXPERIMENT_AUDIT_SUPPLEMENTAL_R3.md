# Experiment Audit Supplemental R3

**Date:** 2026-10-21  
**Scope:** first SSM batch; semantic and batch-loading delta only  
**Reviewer:** GPT-5.6-Sol ultra, same fresh reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Overall verdict:** **WARN (unchanged)**

The two frozen-document `r3` commands passed exactly once without repair, rerun, or execution divergence. This supplemental evidence closes the main all-channel semantic, stored-mirror, manifest-equality, and real-DataLoader coverage gaps. It does not change the algorithmic-proxy classification, unresolved physical source timing, small evaluation scope, subject overlap, lack of generation-time GMR proof, or absence of robot/model-performance evidence.

The original `independent_review/EXPERIMENT_AUDIT.md` and `.json` remain unchanged.

## Frozen inputs

- Processing document SHA256 before and after both commands: `5e1ccc5b0ae9898bb2d32dfdff7116abb77a44f4674ee5faa63cf5af47e3b34a`.
- Claims report SHA256: `11aef5367828c8d1224aca030b21d64d0b02a2219165ba6b48ea334a00ef6855`.
- Semantic checker local/remote SHA256: `6c073d04ce2dcb1fa8cf4ad3076a2c526d0453c37bf5608cb11ceab3107146e9`.
- Batch-loading checker local/remote SHA256: `fb9c3b8032abb7206c825743bf1204f7993489e6f947bba7b806ce7329db871f`.

## Document-following execution

All three new output paths—the semantic receipt, its derived legacy-core receipt, and the batch-loading receipt—were absent before execution.

### Semantic command

- Exact command: `semantics_r3_command.txt`; 981 UTF-8 bytes without newline; SHA256 `a756172c3da1a55df3f6d5175ef5655ae10d4a1ed4561943c7d96932855f58ee`.
- Client UTC: `2026-10-21T13:39:43.8171886Z` to `2026-10-21T13:40:01.9026067Z`; exit `0`.
- Exact stdout: `semantics_r3_stdout.txt`; 3,853 bytes; SHA256 `85d0b182134ce0dd21b67142c9a03b12b496dda7656d33c1c26362af8142cc34`.
- Exact stderr: `semantics_r3_stderr.txt`; 0 bytes; SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Semantic receipt: `ssm_semantics_document_r3.json`; 25,495 bytes; SHA256 `cb924d86939df1d8738aca0c1ff7142b1d130664a2db598f764beb60f1684d23`.
- Derived legacy receipt: `ssm_semantics_document_r3_legacy_core.json`; 883 bytes; SHA256 `b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c`.
- Statuses: `DATASET_CONSISTENCY_CHECKS_PASSED` followed by `SUPPLEMENTAL_SEMANTICS_CHECKS_PASSED`.

The receipt reports all 266 payload files checked, manifest/official-bucket equality, all frames and all 262 channels recomputed, exact contact channels, maximum sequence-mean RIC residual `1.4466670078791968e-08 m` under `2e-05 m`, largest continuous-channel error `9.225268013057786e-06` under `1e-4`, and 33 stored mirror pairs checked. Maximum mirror body-position residual was `9.999965549067902e-06 m` under `2e-05 m`; all other reported mirror residuals were zero.

### Batch-loading command

- Exact command: `batch_loading_r3_command.txt`; 562 UTF-8 bytes without newline; SHA256 `8b12e91b3971e93fbb701ef064e1df819a6a12d9fe00c3b21e5abde912936942`.
- Client UTC: `2026-10-21T13:40:46.2353401Z` to `2026-10-21T13:40:49.0373959Z`; exit `0`.
- Exact stdout: `batch_loading_r3_stdout.txt`; 777 bytes; SHA256 `86aedb0a407d3bd9c83e539dfe0cb2c7fa124f67e9b33201722beaa42e12a619`.
- Exact stderr: `batch_loading_r3_stderr.txt`; 0 bytes; SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Receipt: `ssm_batch_loading_document_r3.json`; 819 bytes; SHA256 `7c1ef57a0ad4c98d45f1464639fa3750b034fb4fc4aa9714f995e20a2de6d59a`.
- Status: `PAIRED_BATCH_LOADING_PASSED`.

The command instantiated `PairedMotionData`, iterated a PyTorch `DataLoader` over all 60 training sequences and 5,650 valid frames in 15 batches of four, and found finite padded tensors and binary valid-contact channels. ASFO and online mirroring were disabled. No generator or optimizer was instantiated; optimizer updates were zero.

Remote receipt times are about 1.8 seconds ahead of client timestamps. This is a clock offset, not a command divergence.

## Prior nine-item closure matrix

| # | Status after R3 | Exact evidence and remaining boundary |
|---|---|---|
| 1. Physical-timebase ambiguity | **Closed as disclosure; scientific ambiguity remains** | `docs/humanml_g1_processing_20261021.md:11` and `scripts/audit_humanml_g1_semantics.py:157-189` explicitly label 13 groups and nominal playback units. Physical source timing remains unsupported. |
| 2. Manifest equality and official split labels | **Closed** | Entry-for-entry accepted/manifest/official-bucket checks at `audit_humanml_g1_semantics.py:85-100`; receipt reports equality. |
| 3. All 262 channels and RIC | **Closed for the explicit independent codec** | All-frame MuJoCo/SciPy reconstruction at `audit_humanml_g1_semantics.py:38-70,101-135`; declared tolerances pass. This is not author-exact codec proof. |
| 4. Base/`M` artifact mirroring | **Closed for stored motion artifacts** | All 33 motion pairs and exact raw-frame intervals checked across root, joints, derivatives, FK positions, and rotations at `audit_humanml_g1_semantics.py:128-156`. Official caption provenance does not constitute an NLP semantic-equivalence judgment. |
| 5. Height heuristic and GMR provenance | **Partially closed** | Formula, 30 recomputed heights, and 63 current GMR source-tree hashes are recorded at `audit_humanml_g1_semantics.py:157-168,189-191`. They are post-run observations, not proof of the generation-time solver tree or retargeting fidelity. |
| 6. Real DataLoader coverage | **Closed** | `PairedMotionData` and PyTorch `DataLoader` are instantiated and fully traversed at `scripts/check_paired_batch_loading.py:23-40`; no training step occurs. |
| 7. Immutable success/failure evidence | **Partially closed** | Semantic checker uses caught failures and atomic rename at `audit_humanml_g1_semantics.py:196-212`. Batch checker remains success-only and non-atomic at `check_paired_batch_loading.py:18-54`; the legacy checker also lacks a structured failure receipt. |
| 8. Effective scope disclosure | **Closed** | `docs/humanml_g1_processing_20261021.md:43` and `audit_humanml_g1_semantics.py:169-188` preserve 2,852 unique canonical frames, two test recordings/87 frames, three identifiers, and two train/test identifier overlaps. |
| 9. Claims-report corrections and preservation | **Closed operationally** | The frozen report fixes proxy, timebase, loader, scope, residual, and test-log language at `reports/HumanML3D到G1配对数据处理报告.md:12,21,26-31,35-45`. This directory now contains both exact command/stream traces, all three byte-matched receipts, execution bindings, and this supplemental verdict. |

## A–F delta

| Check | Original | Supplemental R3 | Basis |
|---|---:|---:|---|
| A. Ground-truth/reference provenance | WARN | **WARN** | Proxy classification and post-run height/source observations are explicit; generation-time GMR provenance and retargeting fidelity remain unproved. |
| B. Score normalization | PASS | **PASS** | No adverse change; train-only normalization remains verified. |
| C. Result/artifact existence | WARN | **WARN** | Full 266-file inventory passes; batch and legacy structured failure records remain incomplete. |
| D. Dead code/check reachability | WARN | **PASS** | Documented commands now reach manifest equality, all-channel/all-frame semantics, screening, all 33 mirror pairs, `PairedMotionData`, collation, and DataLoader traversal. |
| E. Scope | WARN | **WARN** | No validation set, two-recording test scope, subject/session overlap, and 13 ambiguous source clocks remain intrinsic limits. |
| F. Evaluation classification | WARN | **WARN** | Remains `other:mixed_real_source_algorithmic_retarget_proxy`; no robot GT, author-exact codec, tracking, training, or performance evaluation. |

## Frozen claims report

The report at SHA256 `11aef5367828c8d1224aca030b21d64d0b02a2219165ba6b48ea334a00ef6855` agrees with the reviewed first-batch evidence:

- Counts and effective scope are accurate at `:11-21`.
- Physical-timebase limits are accurate at `:26-27`.
- Height heuristic, proxy classification, and provenance boundary are accurate at `:28`.
- Actual DataLoader coverage and its non-training boundary are accurate at `:31`.
- Rounded residuals are corrected at `:35`.
- The `8 + 6` tests are correctly attributed to implementer logs at `:37`; this incremental pass did not independently rerun those tests.
- Supplemental semantics are correctly attributed to the implementer at `:41`.
- Same-family/provisional and non-performance boundaries are retained at `:39,43-45`.

## Remaining WARN boundaries

1. The physical timebase of 13 source groups remains scientifically unresolved; only nominal HumanML indexing/playback is supported.
2. Current GMR tree hashes and recomputed height values do not prove the generation-time source tree or retargeting fidelity.
3. The batch and legacy checkers still lack atomic structured failure receipts.
4. The batch remains a small, no-validation, recording-disjoint but subject-overlapping data-preparation set.
5. G1 targets remain algorithmic proxies. There is no real robot GT, author-exact representation proof, physical tracking, dynamics, generator training, or model-performance evidence.

## Updated claim impact

- **Supported:** complete 266-file inventory; accepted/manifest/official-bucket equality; stored features matching the explicit independent 262D codec within declared tolerances; exact contacts; 33 stored motion-mirror pairs; real `PairedMotionData`/DataLoader traversal; recording-level isolation.
- **Needs qualifier:** 5,904 as repeated/mirrored sequence-frame visits; mirroring as an explicit reconstruction convention; current post-run GMR tree and height observations.
- **Unsupported:** physical 20 Hz for all sources; subject-disjoint generalization; generation-time GMR proof; retargeting fidelity; author-exact 262D equivalence; real robot GT; dynamics; tracking; generator/model performance; complete HumanML3D or PhyGile reproduction.
