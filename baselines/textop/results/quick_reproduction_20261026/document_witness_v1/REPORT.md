# Fresh document-following preflight witness

Result: PASS; no doc-vs-reality divergence observed.

The documented invocation was executed exactly once with the stated Python executable, repository working directory, PowerShell call operator, script, and arguments. Only stdout/stderr capture redirections were appended. No --deploy-disabled, retry, repair, activation, training, source edit, or additional SDK action was performed.

## Exact documented invocation executed

```powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' baselines/textop/provider_tools/quick_budget_preflight.py --run document_preflight_v1
```

Working directory:
C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation

Actual process exit: 0.

Actual stdout (also preserved separately in stdout.txt):
```text
{"status": "QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED", "completed": 2100, "checkpoint_ready": false, "signals_sent": 0}
```

Actual stderr: empty (also preserved separately in stderr.txt).

## Document versus observed receipt

| Check | Document expectation | Actual observation |
|---|---|---|
| Actual process exit | 0 | 0 |
| Preflight status | QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED | matches |
| Completed iterations | below 5000 | 2100 |
| Quick target | 5000 | 5000 |
| Controller enabled | false | false |
| Target checkpoint ready | false | false; checkpoint null |
| Signals sent | 0 | top-level 0; observation empty array |
| New SDK runs | 0 | top-level 0; observation sdk_runs 0 |
| Frozen runtime files | all 16 verified | frozen_runtime_files_verified 16 |
| Transport exit/stderr | clean | 0 / empty |
| Reproduction label | QUICK_REPRODUCTION_NOT_PAPER_COMPLETE | matches |

The receipt additionally records QUICK_BUDGET_WAITING_FOR_5000, 42000 optimizer updates, trainer PID 3812580 and owned group 3812573. This witness verifies the disabled controller and active-state observation only. It does not establish a completed 5000 checkpoint, an actual controlled stop, pilot tracking quality, formal 40000-round completion, or full paper reproduction.

The provider ledger's unchanged warm tracker/resource contract is consistent with the quick document. No environment rebuild or numerical/SDK witness was required or performed by this read-only pass.

Original generated receipt:
C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/results/quick_reproduction_20261026/document_preflight_v1.json

A byte-for-byte copy is retained here as document_preflight_v1.json. Exact invocation, cwd, actual exit, stdout and stderr are saved separately in this folder.
