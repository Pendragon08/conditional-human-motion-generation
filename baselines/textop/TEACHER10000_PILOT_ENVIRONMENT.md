# Real10k teacher paired evaluation runtime

User authorized assessing physical filtering at10k before deciding about40k.
This separate10k schema uses the actual backed-up completed_10000.pt, not a
spoofed5k/40k receipt. The original production run was interrupted only after
backup, using one original-process pidfd SIGINT. Its last published count11400
is retained; only10000/200000 is recoverable here. Original process/supervisor
exit0 and outerexit1 are retained as authorized handoff, not40k completion.

Warm provider ledger SERVER_NOTES.md and FULL_TEACHER_RESUME_ENVIRONMENT.md.
No rebuild/dependencies. SharedGPU1, headless, CPU0-3, Torch.65 are explicitly
authorized; do not require an emptyGPU. Need actual freeGPU1>20GiB, available
RAM and disk checks in unchanged supervision. Only oneSDK/common exclusive lock.
Old57D sleeping/stopped processes remain untouched. SDK GPU0 auxiliary context
is recorded, not claimed absent. Nativeclose fallback remains a difference.

Spec configs/teacher10000-pilot-spec.json SHA
3b85a64c8ef4f9eaaaa068cb65cc3e6560801588c75b421c5242de51ee0bb838:
smoke enabled, sample disabled. Realcheckpoint SHA
6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45.
Sources: teacher10000_pilot_protocol.py, evaluate_teacher10000_pilot.py,
supervise_teacher10000_pilot.py. Frozen old5k executor/protocol/trace remain.
The reset/control/recording core is byte-identical to verified5k except the
RUNNING receipt status string; seed20261026, original strict thresholds,
validated solidfloor and environment sizes all match. Source selection is
also checked against the existing frozen private8/64 selection SHA.

Eight smoke trials are complete52..64frame originals, 8env/1024control calls/
600seconds, followed only after author+fresh actual receipts by64sample16env/
8192calls/3600seconds. Same short8 as5k, not8new independently sampled motions.
Sampling full library is8278 references, evaluation only the selected subset.
No optimizer update is permitted; limited evaluations never accept a whole corpus.
Current-target body/joint errors and terminal reference analysis are separate.
Recording is initial then pre-autoreset physical state; T-1 calls consume0..T-2.
Terminal referenceT-1 may be compared post-hoc at correct elapsed time, not a
newly consumed control target. Failures remain, no best-retry selection.

## Fresh actual document invocation

Use localcwd C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成,
PYTHONIOENCODING=utf-8, local interpreter
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe.
Read/review the code and handoff first. Confirm author teacher10000_smoke_author_v1
has actual outerexit0/child0/closedwriters and no existing document run before launch.
Do not rerun18CPUfixtures, previous5k/10kauthor or any trainer/backup/controller.
Execute exactly once:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/launch_teacher10000_pilot.py' --stage smoke --run teacher10000_smoke_document_v1
```

This returns a durable driver; dispatch is not success. Inspect existing receipts,
worker/logs until bounded completion; never relaunch a run. Read with session.remote:
ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024,
ROOT/tracker_validation/teacher10000_smoke_document_v1/receipt.json,
ROOT/receipts/teacher10000_pilot_teacher10000_smoke_document_v1_{exit,process_exit,outer_exit}.json,
ROOT/logs/teacher10000_pilot_teacher10000_smoke_document_v1_resources.jsonl.
Store rawSDK log/arrays only privately if needed. Failures stay in place, no fixes
or numerical/config changes during this pass.

After exit, use existingROOT/env/bin/python CPU-only to load the current spec,
verify its actual source hashes and load_teacher/authorize_teacher. In an in-memory
copy only, fill smoke_witnesses with the existing author and fresh run names and
actual receipt/exit SHA values, then call verify_smoke_witnesses(ROOT,copy,checkpointSHA).
Do not change spec or activation flags. Require mode teacher10000_pilot,
TEXTOP_TEACHER10000_PILOT_EXECUTION_COMPLETED, 10000/200000,8closed/at least1natural,
0optimizations, privatejournal/index/NPZ/binary SHA equality, all samples finite,
29joint order, consumed sequence initial0 then0..steps-1, and natural duration
steps=T-1. All actual child/outer0 and real resource/process/close proof required.
Full64 comparison/whole-library acceptance/semantic quality remain unclaimed.
If8all succeed do not claim failure branch exercised here.

Review same numerical/selection/core binding versus5k, original20frozen runtime
and handoff/control hashes unchanged, no production worker overlap, and resource
policy. Record native_close_returned/teardown_fallback_used as they actually are.
Write REPORT.md and SUMMARY.json plus small aggregate proof to new local folder
work/textop_reproduction_20261024/teacher10000_pilot_document_v1.
No per-source identity or rawarray in public report. Parent will archive each
closed newrun privately before changing sample activation metadata.
