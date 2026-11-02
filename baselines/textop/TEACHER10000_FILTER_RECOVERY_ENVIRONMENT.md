# Filtering-first continuation after the10k empty relaxed gate

User explicitly requested safely pause the prefilter generator and finish
all-library filtering before generator training. User allowed10k evaluation
before committing to40k. That real10k evaluation is complete: fixed64 gave46
natural/18failed. NEW all8278 saved relaxed statistics contain15887307 attempts,
no unattempted source, but no EMA failure probability<=0.05 (minimum.1866086).
This is not proof every motion is infeasible. Preserve assets/thresholds and
negative results. Pinned release initializes failed/success EMA to1, alpha.001,
stable-count updates on attempted sources; historical statistics include early
learning and initialization. Causality of the empty gate is not established.

Generator is really stopped at8100 updates. Server+D private513526637B snapshot
SHA2b958a5e4efbd74ddfe6bb99b8a5eb8174cec143bb782689b1f6af8934a6c2fe,
CPU1004finite tensors/model/Adam reload/RNG checks passed. Realworker/supervisor/
outerexit1 and staleRUNNINGreceipt are retained, with separate accepted userpause
receiptfba3cef55cfb4afd1cc837d73bb926f3d47838fd4162538d143939ef48d64472.
No further prefilter VAE/LDM. Do not rerun pause/backup or signal oldworkers.

Next productive step is continue original teacher from actual10k fullcheckpoint
toward the existing paper40k budget, reviewing the next real15k checkpoint;40k
completion is not imposed before the next review. Same original20frozen files,
8192env GPUphysics/557D/PPO/rewards/reset/solidfloor; no numeric edit. New separate
supervisor changes only sourcecheckpoint, outputrun, provenance and extra gates.
CPUmetadata preflight alreadypassed. Actual production load_recorded verifies
model/Adam/normalizers/EMA/rawcounts inGPU and fresh episodes restart; RNG/physics
trajectory is not bitwise continuation. This is not accepted filtered data.

Fresh independent reviewer: read new supervisor/spec and compare with original
supervise_full_teacher_resume.py. Check actual realgeneratorpause and10khandoff,
privatefullcheckpoint SHA, all20frozen bindings, originalsourceSHA, relaxedsummary,
no actualVAE/teacher/SDK workers and free commonlock, ownedpersistentdata budget,
GPU1 sharedfree>=20GiB, RAM>=20GiB, CPU0-3/Torch.65. Do not require an emptyGPU.
No package installation, olddiagnostic repeat, config/threshold modifications,
oldrun restart, checkpoint deletion or sourcearray publication.

Local workspace C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成.
Python C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe.
Set PYTHONIOENCODING=utf-8. BASE=work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop.
ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024; use existing
BASE/provider_tools/session.py remote/SSH; new spec initiallyresume_enabledfalse.

After static/read-only checks without blockers, invoke this ONE documented local
command. It preserves disabledspec, runs fresh preflight, enables only resumeflag,
reserves unique driver and starts durable supervised production exactly once:

```
python BASE/provider_tools/launch_teacher10000_filter_recovery.py
```

Expand BASE and use the explicit warm localPython above. Do not run this again
after dispatch or retry on unknownnetworkoutcome. Inspect realdriver/run first.
Run=filter_policy_paper69_resume10000_filter_recovery_20261027_v1.
Expected startup receipt ROOT/tracker_validation/RUN/receipt.json with
RUNNING_FORMAL_FILTER_TRAINING/resume_exact_state_verified=true; receiptcompleted
remains0 until10100 because original100round flushing. Read originalnative log
for actual10000+ iterationfinite losses; don't fabricate receipt counts.
New realSDK process must have correctcommand andsingleactualPythonargv[0], not
count bashwrapper asanotherSDK. GPU0auxiliarySDKcontext record honestly.
Production remains running after review; do not stop it to finish the document.
Check actualexitabsent/lockowned/CPU0-3/frozenfilesunchanged; allpre/postproofs and
REPORT/SUMMARY go to work/textop_reproduction_20261024/teacher10000_filter_recovery_document_v1.
Only bounded aggregate proof public, no fullcheckpoint/privatesource indices.

Shared4cores/GPU1/.65/headless explicitly authorized. Originaltrainer checks
remainingpersistentbudget every100rounds/eachsave; startrequire3892314112B,
actualpreflight5234978816B. Newfuture teachercheckpoint15000/20000 etc must be
backed up using a new run-bound tool (old hardcoded5k/10k backup tools are invalid).
Keep every checkpoint and reserve2GiB+512MiBlog. Subsequent generator/WBC/media
capacity is not solved. At15k inspect saved relaxed statistics and actual teacher
quality before selecting strict all-survivor replay; do not silently loosen5%.
