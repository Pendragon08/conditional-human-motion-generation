# Independent tracker document witness v1

Result: **UNFINISHED / NOT DOCUMENT-VALIDATED**. The single documented PowerShell SSH invocation was executed verbatim once. It emitted `TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED` and wrote the expected receipt, but the direct command remained open after the sentinel. The actual command exit code is **unknown**, not zero. Exec session ID: **85961**.

## Scope and evidence

Only the supplied run-experiment skill, its compute environment contract, SERVER_NOTES.md, and TRACKER_ENVIRONMENT.md were used to orient this fresh witness. No repairs, invocation retry, package/environment changes, training launch, process stop, or publication were performed. The receipt was read over SSH; that separate read exited 0. Licensed private reference data and generated USDs were not copied.

The command's output was captured verbatim in `stdout_stderr.log`. The tool supplies a combined stdout/stderr stream, so it cannot be truthfully split after execution. `execution_record.json` retains command text, every captured exec chunk, and the explicit unknown-exit state.

## Every documented receipt expectation

- status: expected `TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED`; observed `TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED`; MATCH.
- dof: expected `29`; observed `29`; MATCH.
- num_envs: expected `8`; observed `8`; MATCH.
- physics_steps: expected `32`; observed `32`; MATCH.
- optimizer_updates: expected `0`; observed `0`; MATCH.
- physical_filter_executed: expected `false`; observed `false`; MATCH.
- full_filter_training_ready: expected `false`; observed `false`; MATCH.
- ground_present: expected `false`; observed `false`; MATCH.
- tracking_success_evaluated: expected `false`; observed `false`; MATCH.
- body_count: expected `30`; observed `30`; MATCH.
- spec_canonical_hash: expected `893388d9`; observed `893388d9`; MATCH.
- source_sha256: expected `b1bcaee5476f4a8d7c660c411f4f2765c13a08889d6755cc29bf71e7fe3b7593`; observed `b1bcaee5476f4a8d7c660c411f4f2765c13a08889d6755cc29bf71e7fe3b7593`; MATCH.
- urdf_sha256: expected `dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b`; observed `dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b`; MATCH.
- seeded_paper_MLP_forward_backward_passed: expected `true`; observed `true`; MATCH.
- MLP_input_dim: expected `557`; observed `557`; MATCH.

The documented sentinel was observed. The required exit0 was **not observed**. All documented receipt values and the documented canonical/source/URDF hashes match. The observed spec file-byte SHA256 is `eda7e35b754a86952175c6325ffb11e2c0abb40848b1d168edb2350d045b4982`; no expected file-byte hash was published in the supplied tracker documentation, so it is recorded without claiming an independent expected-value comparison.

The receipt also reports finite real state transition max 0.1294969916343689 m, peak Torch allocation 62094336 bytes, and completion UTC 2026-10-25T04:27:42.387894+00:00. The private-reference digest is retained only as receipt metadata. Runtime output names device cuda:1. The receipt does not enumerate Python/package versions, the CPU thread cap, Torch fraction, or MLP hidden-layer widths; the supplied docs' corresponding details are therefore not independently re-certified by this receipt alone.

## Document-vs-reality divergence and runtime diagnostics

1. **Teardown / exit divergence:** Documentation requires the independent direct command to return exit0 before backend document validation. At the last observation the command was still open with no further output after the success sentinel; exit status remains unknown. The parent explicitly ended this bounded witness observation and requested that this divergence be preserved. This witness must not be used to certify exit0 or full environment readiness.
2. Startup did continue and produced a receipt despite logged diagnostics: GLFW initialization failures and failed startup of `carb.windowing-glfw.plugin`; an Error from `gpu.foundation.plugin` saying `DriverShaderCacheManager::init() called without a shutdown()!`; default-display verification failure; key-value database disabled because another kit process locked it; no Viewport Window; OmniHub inaccessible; missing rendering_modes extension config; MaterialX enablement warning; no crash reporter; and the dynamic_control deprecation warning. These are preserved in the actual combined log, not silently treated as clean startup.
3. No contradictory receipt field or documented expected-hash mismatch was observed. The document's limitation is respected: this is a free-fall backend/kinematic numerical witness, not a contact test, success-rate evaluation, physical filter, trained tracker, or full TextOp result.

## Command executed once

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/run_tracker.sh /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/tracker_kernel_witness.py --spec /data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/tracker-env-spec.json --output /data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/document_v1 --device cuda:1 --headless'
```

## Local retained artifacts

- command.ps1: exact documented command.
- stdout_stderr.log: actual combined native output.
- receipt.json: aggregate receipt read from the server.
- execution_record.json: status, check table, and full captured tool chunks.
- receipt_read_record.json: receipt-read command result.
