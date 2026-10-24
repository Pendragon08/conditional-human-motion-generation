# Fresh-reader witness report

**PASS — exact documented invocation executed once, exit 0.** This is the bounded agent-follows-doc pass required by the run-experiment skill. It certifies this simulation execution only.

- Run: 2026-10-22T16:01:08.505721+00:00 to 2026-10-22T16:01:57.098505+00:00 (2026-10-23 00:01:08–00:01:57 Asia/Shanghai).
- Actual physics control calls: **200**. Effective activated replans: **7**, at policy steps **37, 61, 84, 108, 131, 155, 178**. Eight asynchronous plans were submitted; the final submission reached the run limit without activation. Stop reason: `max_policy_steps_reached`.
- All saved numeric arrays are finite: executed qpos [200,36], body positions [200,33,3], timestamps [200], observed prefixes [9,25,36], and all six motion-buffer arrays. Minimum root height: 0.603984312 m.
- First actual CFG eager/compiled comparison: maximum absolute difference **0.00000667572021484375**, with atol=rtol=0.0002; passed.
- Shared GPU1, CPU threads 4, PyTorch allocation cap 16%. Preflight free-memory threshold >8192 MiB passed; the measured free-memory value was not retained. Runtime versions match the document: Python 3.10.21, Torch 2.5.1+cu121, MuJoCo 3.3.7.

| Training rank | Validated updates before | Validated updates after | Advance | Status / skipped |
|---|---:|---:|---:|---|
| 0 | 13975 | 13982 | +7 | TRAINING / 0 |
| 1 | 13975 | 13982 | +7 | TRAINING / 0 |

Both snapshots retained validated optimizer and EMA states, zero offending parameters and zero consecutive skips. Full training remains incomplete.

Source and checkpoint identities:

- Official source commit: `5a79d03cab824eb5917271ba720294e8a1b9d374`.
- Runner `code_v4/run_reactive_async.py` SHA-256: `2728711d6ea5fb8f371ae5970f01adcb12c4f1196f7e9e334ec0df0368731df1`.
- All **119** recorded source hashes were unchanged. Complete path/hash map is in `result.json → receipt.source_hashes`.
- Planner `model000010040.pt` SHA-256: `7aaed03598e120e843b4332b5943605a293bbc31958e05b4396ec791c083dbe5`; unchanged before/after.
- Policy `ScaleBFM_m_model_22200.pt` SHA-256: `88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3` (72856655 bytes). This is a post-run identity check; the wrapper's before/after checkpoint assertion covers the planner checkpoint.

Document-versus-reality finding: the original provider-ledger reference `../reactivebfm/docs/recovery_provider_ledger_v3.md` was absent. The actual ledger was found in the separate ReactiveBFM checkout and read before execution. The author then corrected ENVIRONMENT.md to that verified absolute path. The fresh reader reread the corrected document after the run; the exact invocation was unchanged and was not repeated. No other execution-expectation mismatch was observed.

Remote raw stdout/stderr are retained at `/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/doc_witness.stdout.txt` and `doc_witness.stderr.txt`; full receipt and training snapshots are in `result.json`. Their hashes, finite-array details and full training counters are in `witness_report.json`. Frame-overrun warnings and an unavailable optional OpenGL_accelerate module did not prevent completion; no real-time performance claim is made.

No implementation edit, dependency installation, process stop, deletion or further experiment was performed by this witness. This pass does not establish semantic quality, paper success rates, full data coverage or completed training.

