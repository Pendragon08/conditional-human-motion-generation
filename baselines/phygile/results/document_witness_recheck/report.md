# Document-following recheck after the final ledger edit

Date: 2026-10-17. Environment document spec hash: fa5883c8.

The independent witness agent read the updated baselines/phygile/docs/environment.md and followed the already-read run-experiment skill. No code, dependency, configuration, or documentation edits were made. Existing jobs were left running. Previous document_witness records were preserved.

## Commands and observed results

The same authorized SSH host and key were used with BatchMode=yes and ConnectTimeout=12. Both invocations below were executed verbatim.

1. `cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017 && bash scripts/run_isaac.sh scripts/kernel_witness.py`
   - Exit status: 0.
   - Seed 42; shape 128 x 128; loss 254.7862548828125; gradient_finite=true.
   - Python 3.10.15; torch 2.5.1+cu118; CUDA 11.8; NVIDIA RTX A6000.
   - Tracking package resolved to `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/upstream/source/whole_body_tracking/whole_body_tracking/__init__.py`.
2. `cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017 && bash scripts/smoke.sh`
   - Exit status: 0.
   - Started 2026-10-17 09:44:52.5348044 UTC; finished 09:45:14.8143981 UTC (about 22.3 seconds including SSH).
   - Observed cuda:0 and 32 environments.
   - Completed Learning iteration 0/1 using MoEPPO, with 768 total timesteps.
   - Collection 4.485 seconds; learning 0.321 seconds; mean value loss 0.9102; mean surrogate loss 0.1477; mean reward -0.53.
   - Runtime also reported: `[Reproduction] Verified joint/body order and 50 Hz timing for 1 references.`
   - New checkpoint confirmed by a separate read-only artifact listing: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_09-45-02_smoke/model_0.pt`, 67,520,342 bytes, modified 09:45:13.644501555 UTC during this run. The checkpoint was not deserialized by this witness.

## Explicit document-versus-reality comparison

- No observed mismatch in documented command syntax, work directory, current-checkout package resolution, GPU selection, environment count, update count, or successful kernel/simulation behavior.
- No pip installation or environment override was needed.
- GPU 0 preflight usage was 5729 MiB / 49140 MiB, and GPU 1 usage was 9732 MiB / 49140 MiB. Another evaluation was running, as communicated by the parent. No existing job was stopped or changed; the smoke succeeded alongside the existing allocations.
- The document still omits non-blocking runtime diagnostics: headless display/GLFW and viewport warnings, shared Kit database lock warnings, extension/shader warnings, git metadata warnings, and `[gpu.foundation.plugin] DriverShaderCacheManager::init() called without a shutdown()!` on stderr. Their presence is an output detail missing from the ledger, not an execution mismatch: the update and checkpoint succeeded and the process exited 0.
- The updated document states that Isaac Lab 2.1.0 was additionally verified from VERSION. This recheck did not repeat that separate VERSION-file inspection, so that statement is not independently re-established by these two invocation outputs. No contrary version evidence was observed.

## Evidence files

Full stdout, full stderr, exact remote invocations, and exit statuses are stored separately using `preflight`, `kernel`, `smoke`, and `checkpoint_listing` prefixes in this directory. No detached process or process termination was used. The simulation finished within three minutes.

Outcome: the final documented commands pass again, including a real PPO update and a newly persisted checkpoint. This check establishes smoke execution, not converged policy quality or full paper reproduction.
