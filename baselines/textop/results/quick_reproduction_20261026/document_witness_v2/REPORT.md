# Fresh document witness v2

Result: PASS_WITH_DOCUMENT_PATH_AMBIGUITY. The documented quick-budget preflight was invoked exactly once, with the documented bundled Python, script, arguments, repository-root working directory, and PowerShell call operator. No retry, deployment flag, activation, repair, process signal, or additional SDK invocation was performed by this witness.

Actual exit: 0. Standard output:

```json
{"status": "QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED", "completed": 2300, "checkpoint_ready": false, "signals_sent": 0}
```

Standard error: empty (0 bytes). Exact command and working directory are retained in exact_command.txt, executed_invocation.txt, and execution.json; the separate captured streams are stdout.txt and stderr.txt.

## Document versus reality

| Document expectation | Observed receipt / execution | Result |
| --- | --- | --- |
| Actual exit 0 / verified status | Exit 0 / QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED | Pass |
| All 16 frozen runtime hashes checked | frozen_runtime_files_verified = 16 | Pass, as recorded by preflight |
| No signals | signals_sent = 0; observation.signals_sent = [] | Pass |
| No new SDK runs | new_sdk_runs = 0; observation.sdk_runs = 0 | Pass |
| Budget disabled | observation.enabled = false | Pass |
| Teacher below 5000 | current_completed_iterations = 2300; quick_target_iterations = 5000 | Pass |
| Target checkpoint unavailable | target_checkpoint_ready = false; checkpoint = null | Pass |
| No destructive work | data_deleted = false | Pass |

The earlier 2100-iteration author observation in the document is historical; the fresh 2300-iteration observation is consistent with ongoing training and the below-5000 expectation.

One documentation path ambiguity: the guide specifies repository-root cwd and then names results/quick_reproduction_20261026/document_preflight_v2.json without its baseline prefix. That repository-root path does not exist. The actual receipt is baselines/textop/results/quick_reproduction_20261026/document_preflight_v2.json. This witness inspected that baseline receipt and saved an exact copy here. No behavioral divergence was observed.

Read scope was limited to QUICK_REPRODUCTION_ENVIRONMENT.md, SERVER_NOTES.md, the requested output receipt, and this witness's own records. All 19 recorded checks passed. Frozen-file verification is evidenced by the preflight receipt; no source review or separate server hash audit was performed.
