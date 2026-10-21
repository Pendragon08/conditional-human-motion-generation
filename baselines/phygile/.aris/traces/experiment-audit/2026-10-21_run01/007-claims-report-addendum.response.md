## Report addendum

File reviewed: `reports/HumanML3D到G1配对数据处理报告.md`  
SHA256: `9addfe1ec5f4d3b9a22760904bddec73e0b96b0b89b8dd2734ea0f089fd66d91`

The report introduces no new dataset-integrity issue. **A–F statuses remain unchanged: A WARN, B PASS, C WARN, D WARN, E WARN, F WARN; overall WARN, same-family/provisional.**

Supported report claims:

- Counts and scope at `:11-21` match the authoritative batch: 30 recordings, 60 IDs, 66 entries, 5,904 sequence-frame visits, 182 captions, 33 mirrored entries, 60/0/6 split, zero rejects, and 266 files totaling 9,419,016 bytes.
- Provenance, marker identity, missing-marker visibility, and header distributions at `:25-27` match primary files.
- Mirroring and authorship boundaries at `:29-30,35,39` are substantially correct.
- The clarified wording at `:35` correctly identifies the first result as the **implementer’s** consistency audit and avoids presenting its residuals as tracking or generation quality.
- The frozen-r2 claim at `:37` is verified: exit 0, empty stderr, and identical before/after document hash `52bbcc...`.
- `results/humanml_g1_pipeline_20261021/independent_review/document_audit_frozen_r2_execution.json:2-25` contains the verified execution metadata and marks the review `same-family` and `provisional`.
- Pre-freeze and frozen-r2 files coexist, so `:37` is correct that the old trace was retained.

Report-specific corrections required:

1. **Line 31 overstates loader coverage.** The audit calls `train.load_manifest`, but does not instantiate `PairedMotionData` or a PyTorch `DataLoader`. Replace “实际通过现有生成器的数据加载器” with “实际调用并通过生成器的manifest加载与输入校验函数”.
2. **Line 35 should use the recorded values.** Replace approximately `3.64e-8` and `1.39e-7` with `3.63e-8` and `1.38e-7`; the exact receipt values are `3.6312626683354665e-08` and `1.3802523524120147e-07`.
3. **Line 37’s file-reference wording is too narrow.** The JSON contains hashes and execution metadata, while the literal command and exact streams are separate companion files. Refer to the `independent_review/` execution JSON **and companion command/stdout/stderr/receipt files**.
4. **Line 37 currently points to an absent “独立审查报告.”** At inspection time that directory contained execution artifacts only. Add the review only after its file exists, or remove the dangling reference.
5. **Lines 26-27 need the physical-timebase qualifier.** For 13 recordings, only nominal HumanML indexing is established; physical 20 Hz and physical speed units remain unresolved because the two headers conflict.
6. **Lines 12 and 28 should make the proxy classification explicit.** Prefer “官方M前缀镜像ID” over “真实镜像ID,” and state that G1 targets are algorithmic retargeting proxies rather than measured robot ground truth. Line 28 should also disclose the `1.66 + 0.1 * beta[0]` height heuristic.
7. **Lines 17, 21, and 31 should disclose effective evaluation scope:** 2 test recordings, 87 unique canonical test frames, no validation data, and both test subject/session identifiers also appearing in train. Recording-disjoint is supported; subject-disjoint is not.
8. The “8 + 6 tests passed” statement at `:37` is not established by the frozen-r2 invocation. The reviewed source contains six structural tests, but the document audit did not run them and no corresponding test log was in the reviewed trace. Cite a preserved execution record or label this as an implementer-reported result.

These corrections refine the existing must-fix items; they do not change any A–F status or claim-impact classification.
