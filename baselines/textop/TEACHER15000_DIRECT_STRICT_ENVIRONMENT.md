# Frozen15000 direct complete physical replay

The user explicitly requested "直接进行下一步物理筛选，不要再训练跟踪器了。".
The recovery teacher has actually stopped, its original process reaped, and the
common SDK lock released. Use only the saved15000 checkpoint SHA
66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88.
Do not resume/stop the teacher again or run the old5k/10k/40k launchers.

This is a new user-directed route: all8278 originals receive one deterministic
strict complete-trajectory replay. Historical EMA<=5% candidates remain0; the
direct route does not assert that the paper's earlier relaxed gate passed.
Keep .25m pelvis and .5m end-effector termination thresholds, orientation check,
the validated solidfloor contact profile, 50Hz control/4physics steps, full T-1
transitions and failure precedence. Record actual poses before autoreset.
Current-target errors are not independent final-endpoint or semantic metrics.
No optimizer updates, corpus acceptance or generator start occurs in this entry.

## Warm environment and scope

Workspace C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成.
BASE is work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop.
Server ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024; uid1013.
The established tracker environment canonical hash b7a3bf0c is unchanged.
Use ROOT/scripts/run_tracker.sh (SDK Python3.10/Torch2.5.1), CPU0-3/4threads,
headless GPU1/Torch .65; SDK's GPU0 auxiliary context remains disclosed.
Shared GPU1 is authorized, so do not require an empty card. Supervisor checks
available GPU1>20GiB, host RAM, free disk and exclusive common SDK lock for each
new job. No installs, shared dependencies or other users' processes are touched.
The old22 production file bindings remain frozen and are checked at launch.

New files: evaluate_teacher15000_strict.py, teacher15000_strict_protocol.py,
direct_strict_trace.py, supervise_teacher15000_strict.py and
configs/teacher15000-direct-strict-spec.json. The spec binds every source hash.
The new compact private format stores all executed36D poses as little-endian
float32; samples=control_steps+1. Each finished trace has exact byte count and
SHA plus strict trial metadata. Partial binaries/metadata are retained after
failure. No duplicate full-library NPZ is created. A new local CPU fixture
checked complete/failure, finite values, budget enforcement and incomplete
preservation; this fixture is explicitly synthetic, not physics evidence.

## Fresh agent follows document (one actual8-source smoke)

First read the new spec and author progress in
results/teacher15000_direct_strict_20261027/teacher15000_smoke_author_v1/progress.json.
Proceed only when completed_execution_verified=true, common_lock_free=true,
actual outer exit0 and no worker remain. If missing/pending, inspect or wait;
do not launch a second overlapping SDK. Read the three new Python sources to
check that the executable behavior agrees with this document.

In the workspace PowerShell, invoke exactly once:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/launch_teacher15000_strict.py' --stage smoke --run teacher15000_smoke_document_v1
```

Wait about45seconds, then collect actual server evidence:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_teacher15000_strict.py' --run teacher15000_smoke_document_v1 --evidence document_v1
```

If still pending, wait30seconds and use the same collect command with the next
unused --evidence document_v2 (then v3, etc). Do not rerun launch after an
unknown client outcome. The unique client intent/outcome/stream files remain
under the local run directory; durable server driver/supervisor proofs decide
the actual outcome. Observe existing server state before any recovery.

Acceptance is real execution and closed outputs:8original sources, all8trials
finished (complete or failed), zero optimizer updates, actual process/outer
exit0, common lock released, finite36D poses/normalized root quaternions,
samples and private hashes exact, journal and final trials identical,
4physics steps/control call and preserved current15000 provenance. At least
one natural completion is required to exercise the completion branch. Each
failure stays a failure. This smoke is not full-library filtering.
Use the collector's actual validation and inspect progress.json for divergence.
Do not modify sources/spec, start full scope, send signals or notify third parties.
Save REPORT.md and SUMMARY.json under
work/textop_reproduction_20261024/teacher15000_strict_document_v1, with exact
commands, exits, evidence SHA, findings and limits. Return findings to parent.

## Full scope after author and fresh witnesses

Only parent activates full_launch_enabled after verifying both actual smoke
receipts/exits and unchanged executable/settings hashes. Full uses16envs and
all8278 originals sorted by (frame count, original global index), independent
of outcome, to reduce padding. It keeps the same numerical replay behavior.
Raw qpos cap1GiB, metadata/log allowance .5GiB, retained disk reserve2GiB;
preflight requires3.5GiB. Overall timeout24h and max2million vectorized calls
are upper safety bounds, not expected duration. Progress is written every50
calls or batch end, resource samples every15s. All original/failed/unattempted
motions remain preserved. A successful full execution is still followed by
independent trial/corpus/text/split/CLIP/capacity acceptance before generation.
