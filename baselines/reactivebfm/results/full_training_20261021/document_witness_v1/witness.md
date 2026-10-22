# Independent document runtime witness

**DOCUMENT_WITNESS_PASSED — 24 checks passed.** The frozen command in `docs/full_runtime.md` was executed verbatim exactly once after an immediate output-absence check. It exited **0** with `FULL_RUNTIME_WITNESS_PASSED`. No remote setup, repairs, retries, package/source/config changes, or unrelated process signals were performed.

Controller UTC: **2026-10-21T16:06:31.4924182Z** to **2026-10-21T16:07:22.2877778Z**. The exact command, byte-preserved stdout/stderr and exit status are in `documented_run.invocation.json`, `documented_run.stdout.txt` and `documented_run.stderr.txt`. Remote preflight UTC was 16:06:33Z during controller interval 16:06:30.5126009Z–16:06:31.4817076Z. Both clock records are retained; cross-clock ordering must not be interpreted as runtime ordering.

| Document requirement | Comparison with individual generated receipts | Result |
| --- | --- | --- |
| Both DDP ranks complete four updates; global batch 1024 | Both `PROBE_COMPLETED`, exit 0; optimizer updates 4; official step 0→4; local batch 512 × world 2. Each rank's configuration equals the frozen JSON. | Pass |
| All three dataset components | Each rank reports AMASS 1403, 100style 6268 and Kungfu 687 entries. | Pass |
| FP32 gradient parity; atol 1e-6 / rtol 1e-5; ≥200 nonzero tensors | 365 compared/nonzero tensors; maximum absolute gradient error 0; required tolerances; 37 deterministic gate/output parameter perturbations. | Pass |
| Finite/upright static tracker for 8 seconds | 400 controls; 7.999999999999852 simulated seconds; finite and upright height proxy true; minimum root height 0.788038 m; reference forcing false. | Pass |
| 100 closed-loop controls / 400 MuJoCo integrations; ≥2 planner calls | Policy calls 100; actual integration counter 400; 2 simulated seconds; 12 planner calls. Eleven replanning events at policy steps 9–99 agree with 12 calls including the initial plan; 12 unique state-prefix hashes; real feedback; no reference forcing. | Pass |
| Source/config/doc preservation | Zero before/after changes: local 13 supplementary files, 226 vendor files, 3 configs, 2 docs and ledger; remote 10 supplementary files, 226 vendor files and 3 configs. | Pass |

All 239 deployed file hashes match local bytes. Three local helpers have no deployed counterpart: `scripts/audit_official_data.py`, `scripts/build_overlay.py` and `scripts/verify_github_backup.py`. There are no cross-host hash mismatches. Bytecode/cache files were excluded from source manifests. Per-set digests and comparisons are in `witness.json`; raw before/after manifests are retained.

Copied **21 generated JSON/text files**, including both rank receipts, metrics, per-phase commands/stdout/stderr, parity/hold/loop receipts and closed-loop event metadata. `copied_evidence.json` records their hashes. No checkpoint or pose binary was copied; the two generated NPZ files appear only in the remote inventory. Main receipt SHA256: **055984e0d1c1f8fcca4d4d79adf2ef39f226701ca5d579f64d1f8b0a2a1bdb29**.

This establishes execution only. The configured 1,000,000 updates and 400,000-step warmup are incomplete. One-batch FP32 parity does not establish BF16/full-training equivalence. The old 101-update pilot falls (minimum root height 0.147663 m). Frame-overrun warnings and 0.36–0.56-second planner calls support no real-time claim. Static uprightness is a height proxy, and its receipt has no separate integration counter. No paper metric or trained-planner quality is reproduced. Existing processes were not signalled; unrelated-job PID continuity and package versions were not independently re-audited.

Evidence note: the auxiliary `capture_summary.json` run array includes two serialized void-return placeholders; `documented_run.invocation.json` is authoritative. This affects local evidence formatting only; there was no runtime retry.