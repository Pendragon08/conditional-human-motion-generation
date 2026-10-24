## Incremental SSM integrity verdict

**Overall remains WARN, same-family/provisional.** Both frozen-r3 commands passed without divergence. The supplemental evidence closes the main semantic-check and real-DataLoader coverage gaps, but it does not change the proxy nature, timebase limits, small evaluation scope, or lack of robot/model-performance evidence.

### Deterministic execution summary

Frozen document SHA256 was `5e1ccc5b0ae9898bb2d32dfdff7116abb77a44f4674ee5faa63cf5af47e3b34a` before and after both commands.

| Check | Result |
|---|---|
| Semantic command | Exit 0; stdout 3,853 bytes, SHA256 `85d0b182134ce0dd21b67142c9a03b12b496dda7656d33c1c26362af8142cc34`; empty stderr |
| Semantic receipt | 25,495 bytes; SHA256 `cb924d86939df1d8738aca0c1ff7142b1d130664a2db598f764beb60f1684d23` |
| Derived legacy receipt | 883 bytes; SHA256 `b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c` |
| Batch-loading command | Exit 0; stdout 777 bytes, SHA256 `86aedb0a407d3bd9c83e539dfe0cb2c7fa124f67e9b33201722beaa42e12a619`; empty stderr |
| Batch-loading receipt | 819 bytes; SHA256 `7c1ef57a0ad4c98d45f1464639fa3750b034fb4fc4aa9714f995e20a2de6d59a` |

New source hashes match locally and remotely:

- `scripts/audit_humanml_g1_semantics.py`: `6c073d04ce2dcb1fa8cf4ad3076a2c526d0453c37bf5608cb11ceab3107146e9`
- `scripts/check_paired_batch_loading.py`: `fb9c3b8032abb7206c825743bf1204f7993489e6f947bba7b806ce7329db871f`

Author and independent semantic values agree. Their receipt/stdout hashes differ only because output paths and timestamps differ. Batch receipts are byte-identical. The server clock is approximately 1.8 seconds ahead of the client clock; this explains receipt-internal end times and mtimes following client-side end timestamps.

### Prior nine-item closure matrix

| # | Status | Evidence |
|---|---|---|
| 1. Physical-timebase ambiguity | **Fully closed as a disclosure requirement; scientific ambiguity remains** | The frozen document explicitly limits 20 Hz and speed values to nominal playback at `docs/humanml_g1_processing_20261021.md:11`; the receipt records 13 ambiguous groups and nominal units through `audit_humanml_g1_semantics.py:157-189`. Physical source timing remains unsupported. |
| 2. Manifest equality and split labels | **Fully closed** | Both manifests are compared entry-for-entry with accepted entries and official buckets at `audit_humanml_g1_semantics.py:85-100`; receipt reports equality. |
| 3. All 262 channels and RIC | **Fully closed for the explicit independent codec** | Independent MuJoCo/SciPy reconstruction covers every frame/channel at `:38-70,101-135`. Largest continuous-channel error was `9.225268013057786e-06`, below `1e-4`; contacts matched exactly; maximum sequence-mean RIC residual was `1.4466670078791968e-08 m`, below `2e-5 m`. This does not establish author-exact equivalence. |
| 4. Base/`M` artifact mirroring | **Fully closed for stored motion artifacts** | All 33 pairs are matched by motion ID and exact raw-frame interval and checked across root, joints, derivatives, FK positions, and rotations at `:128-156`. Maximum body-position residual was `9.999965549067902e-06 m`, below `2e-5 m`; all other reported mirror residuals were zero. Official `M` text remains bounded to official-caption provenance rather than an NLP judgment of semantic equivalence. |
| 5. Height heuristic and GMR provenance | **Partially closed** | Formula, 30 recomputed per-source values, and 63 current GMR source-tree hashes are recorded at `:157-168,189-191`. The receipt correctly labels them post-run observations; they cannot prove the exact generation-time solver tree or retargeting fidelity. |
| 6. Real DataLoader coverage | **Fully closed** | `PairedMotionData` and PyTorch `DataLoader` are instantiated and traversed at `check_paired_batch_loading.py:23-40`: 60 sequences, 5,650 valid frames, 15 batches; ASFO/mirroring off; no generator or optimizer. |
| 7. Immutable success/failure evidence | **Partially closed** | The semantic checker catches failures and atomically renames a temporary receipt at `audit_humanml_g1_semantics.py:196-212`. The batch checker still writes success only and has no structured failure receipt or atomic commit at `check_paired_batch_loading.py:18-54`. |
| 8. Effective scope disclosure | **Fully closed** | Exact scope is now stated at `docs/humanml_g1_processing_20261021.md:43` and independently emitted at `audit_humanml_g1_semantics.py:169-188`: 2,852 unique canonical frames, two test recordings/87 frames, three identifiers, and two train/test identifier overlaps. |
| 9. Claims-report corrections | **Partially closed pending trace persistence** | The frozen report fixes values, proxy language, timebase, loader wording, scope, and test-log attribution at `reports/HumanML3D到G1配对数据处理报告.md:12,21,26-31,35-45`. At inspection, `independent_review/supplemental_r3/` existed but contained no files, so the preservation claim at report line 43 becomes true only after the executor saves these execution records and supplemental verdict there. |

### A–F delta

| Area | Prior | Supplemental | Reason |
|---|---:|---:|---|
| A. Ground-truth/reference provenance | WARN | **WARN** | Proxy classification is now explicit, and post-run height/source observations improve traceability; generation-time GMR provenance and retargeting fidelity remain unproved. |
| B. Normalization | PASS | **PASS** | No adverse change; train-only statistics remain verified. |
| C. Result/artifact existence | WARN | **WARN** | Full 266-file inventory verification now passes, but failure-record coverage remains incomplete for the batch checker and legacy checker. |
| D. Dead code/check reachability | WARN | **PASS** | Manifest equality, all-channel semantics, all-frame FK, screening, 33 mirror pairs, `PairedMotionData`, collation, and DataLoader iteration are now reached by documented commands. |
| E. Scope | WARN | **WARN** | No validation set, two-recording test scope, subject/session overlap, and 13 ambiguous source clocks remain intrinsic limits. |
| F. Evaluation classification | WARN | **WARN** | Remains `other:mixed_real_source_algorithmic_retarget_proxy`; no robot GT, author-exact codec, tracking, training, or performance evaluation. |

### Frozen claims report

The report matched its expected SHA256:

`reports/HumanML3D到G1配对数据处理报告.md`  
`11aef5367828c8d1224aca030b21d64d0b02a2219165ba6b48ea334a00ef6855`

Its first-SSM-batch claims now agree with the reviewed evidence:

- Counts and effective scope are accurate at `:11-21`.
- Physical-timebase limits are accurate at `:26-27`.
- Height heuristic, proxy classification, and provenance limit are accurate at `:28`.
- Actual DataLoader coverage and its non-training boundary are accurate at `:31`.
- Exact rounded residuals are corrected at `:35`.
- The `8 + 6` tests are correctly attributed to implementer logs at `:37`; these were not independently rerun in this pass.
- Supplemental semantics are accurately attributed to the implementer at `:41`.
- Same-family/provisional and non-performance boundaries are retained at `:39,43-45`.

The only current report-specific correction is operational: populate `independent_review/supplemental_r3/` with the two command traces, receipts, and supplemental audit before treating line 43’s preservation statement as satisfied.

### Updated claim impacts

- **Supported:** complete 266-file inventory; manifest equality; stored features matching the explicit 262D codec semantics within declared tolerances; exact contacts; 33 stored motion-mirror pairs; real `PairedMotionData`/DataLoader traversal; recording-level isolation.
- **Still needs qualifier:** 5,904 as repeated/mirrored sequence-frame visits; mirroring as an explicit reconstruction convention; post-run GMR source/height observations.
- **Unsupported:** physical 20 Hz for all sources; subject-disjoint generalization; generation-time GMR source proof; retargeting fidelity; author-exact 262D equivalence; real robot GT; dynamics, tracking, generator quality, paper performance, or complete HumanML3D/PhyGile reproduction.
