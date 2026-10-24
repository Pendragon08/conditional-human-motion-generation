# Fresh agent-follows-doc contact witness v4

Verdict: **PASS**. The exact documented PowerShell command was invoked once through a UTF-8 WITH BOM .ps1 file. Actual top-level exit code: 0.

Eight actual CPU MuJoCo-Warp first episodes were evaluated with seed 2026 and no optimizer updates. No dependency installation, source repair, retry, server action, or ReactiveBFM process action was performed.

| Profile | Clip | Frames / motion | Physics steps | Full clip success | MPJPE m | MPJAE rad | MPJVE rad/s | Foot coefficient |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| official | 0 | 129 / 149 | 516 | False | 0.10080162 | 0.14837064 | 0.59854758 | 0.6 |
| official | 1 | 10 / 149 | 40 | False | 0.07968477 | 0.16576977 | 1.65620983 | 0.6 |
| official | 2 | 149 / 149 | 596 | True | 0.08948105 | 0.13402718 | 0.42945704 | 0.6 |
| official | 3 | 14 / 149 | 56 | False | 0.05893192 | 0.15128413 | 1.06564057 | 0.6 |
| matched_sliding_friction | 0 | 129 / 149 | 516 | False | 0.10574293 | 0.14980905 | 0.55782282 | 1.0 |
| matched_sliding_friction | 1 | 9 / 149 | 36 | False | 0.07400952 | 0.15670326 | 1.73549986 | 1.0 |
| matched_sliding_friction | 2 | 149 / 149 | 596 | True | 0.10137459 | 0.13043693 | 0.39781368 | 1.0 |
| matched_sliding_friction | 3 | 15 / 149 | 60 | False | 0.06096819 | 0.15533586 | 1.01383507 | 1.0 |

All acceptance checks: True. Pre/post source-and-input snapshot files: 382. Unique executed source/input files independently rehashed: 248.

Broad directory snapshots differed only by one newly created scripts/after_training.py that was absent from all executed-source manifests. Every pre-existing tracked source and explicit input remained unchanged. This unrelated addition is preserved in command_receipt.json and source_hashes_after.json.

Early termination is permitted by the diagnostic acceptance condition. Metrics are censored at the first termination and must be read with episode duration. These are four development clips, not a held-out benchmark. No performance improvement is required or inferred.

The witness retains exact commands, top-level and child stdout/stderr, actual exit codes, copied receipts and executed-source manifests, before/after source hashes, and independently recalculated acceptance. Private raw motions and recorded states remain only in the original private comparison directory.

Doc-versus-reality: no invocation, path-resolution, exit-status, friction, finite-metric, or executed-source mismatch was found. Child stderr retained a non-blocking Torch torch.jit.script deprecation FutureWarning.
