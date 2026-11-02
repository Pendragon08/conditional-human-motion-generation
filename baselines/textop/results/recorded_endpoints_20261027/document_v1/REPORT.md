# Fresh document-following validation: recorded endpoint audit

**Result: PASS, no observed document-vs-reality divergence.** A fresh agent followed `RECORDED_ENDPOINT_ENVIRONMENT.md` without changing its invocation, runtime, or spec. The exact documented Python snippet was extracted and executed once using the documented local interpreter and working directory, with `PYTHONIOENCODING=utf-8`. Both local and remote exits were 0; stderr was empty; no timeout occurred. Elapsed wrapper time was 2.407 seconds. 43 explicit evidence checks passed.

## Evidence and reproducibility

- Document SHA256: `d8f34ddd3677ab031864d5d435b4264e87fcc7ac4dbff0dbee3dcbd771f70f10`.
- Invocation SHA256: `34431aa0ad91b445e3cc994c5c3bbe092e484d7887012b8dd80151405b70c398`.
- Spec SHA256: `6db8a9171e8ed95cba7db0a31a3ad73cfe0be32f8fca0eff51d84596796d400b`.
- Independently verified server summary SHA256: `9f635efe3f0bf8add73cf4a944b031927aaba6b4d28762ebb2416ec6597ad315`.
- Returned sentinel: `EXISTING_ENDPOINT_AUDIT_INDEPENDENTLY_RECOMPUTED`, matched trials: **64**.
- Four server-bound source files match the spec and the local files actually reviewed. The original recording executor also matches the SHA in the immutable original receipt.
- All **16** frozen training runtime files still match their original quick-plan hashes. The quick-plan SHA remains bound to the original value.

`documented_invocation.py` is the exact snippet. The stdout/stderr files and `invocation_proof.json` retain its result. `read_only_evidence.json` and `additional_read_only_proof.json` contain only aggregate evidence and file hashes. `SUMMARY.json` records every check. No raw arrays or per-source identities were copied locally or published.

## Code and definition review

The auditor verifies the original receipt, real exit/supervision result, process/resource receipts, strict decision log and journal, 64 selected source identities, 8278-source reference fingerprint, exact joint/body names, full reference-file hashes, trace binary/NPZ digests, and binary-to-NPZ array equality. Its verification branch recomputes the private metric JSON bytes and the complete summary in memory, checks exact equality with the saved outputs, then returns before any output-directory creation or write.

Time indexing is consistent with the document: a T-frame motion has T-1 physical transitions, and the retained sequence includes the initial sample. The recorded consumed-target sequence is `[0] + range(control_steps)`. Each of the **42** completed trials has the actual terminal state at elapsed `(T-1)/50`, compared independently against reference frame `T-1`; its last consumed control target is `T-2`. No control call or integration step was added. The old receipt still has `final_reference_endpoint_comparison_performed=false`, preserving the distinction between original execution and later analysis.

The original executor records root position after subtracting the scene origin. Terminal metrics directly subtract the independent reference pelvis position, with no per-frame yaw/root alignment and no initial-offset subtraction. Joint names are unique, sets must match, and the reference joint columns are reordered to the recorded joint order before subtraction. The orientation metric normalizes the two wxyz quaternions and uses `2*acos(abs(dot))`, making quaternion sign immaterial. Root norm/horizontal norm/absolute height error, joint RMSE and maximum absolute joint error match their documented definitions.

All **22** failed trials actually ended strictly before the nominal final elapsed time. They keep their strict failure outcomes, have no terminal metric fields, and are excluded only from terminal aggregate statistics; no missing endpoint is extrapolated. Simultaneous termination wins over natural completion in the reviewed strict coverage code. No acceptance threshold is added or changed.

| Terminal metric over 42 completions | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Root position error (m) | 0.160800465 | 0.090818640 | 0.527158541 | 1.357581175 |
| Joint RMSE (rad) | 0.175055566 | 0.144504580 | 0.307674343 | 0.884651988 |

The maximum initial root offset across all 64 retained records is exactly **0 m**. The documented means of approximately 0.1608 m and 0.1751 rad agree with the recomputed summary.

## Execution scope and limits

The exact invocation uses the existing server interpreter, empty CUDA visibility, and one OMP/MKL/OpenBLAS thread. The auditor enforces UID 1013 and CPU affinity 0–3. Its imported source closure contains NumPy and standard-library helpers, with no Torch/SDK import. The trace class has writer methods, but the verification path imports only its constants and never instantiates it. No signal API or training process control exists in this path. No SDK/GPU job, optimizer, model load, physical trial, checkpoint, prior pilot, or CPU fixture was launched; no runtime/spec/production document was repaired or edited. Only the requested fresh local report directory was written.

This is validation of existing 5k endpoint analysis and its instructions. It does not establish semantic quality, whole-trajectory quality, paper benchmark performance, accepted corpus completeness, or a 10k result. Trainer/coordinator liveness was not independently measured here; the code/frozen-file checks show no runtime edit but are not a liveness certification. Resource settings and absence of control operations were reviewed in source; no new profiler/kernel witness was introduced.

No documented invocation failure or observed definition mismatch required a repair. Existing failed trials and historical failures remain intact.
