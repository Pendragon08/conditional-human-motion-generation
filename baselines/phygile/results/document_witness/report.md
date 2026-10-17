# Fresh-agent document witness

Date: 2026-10-17. Environment document spec hash: fa5883c8.

The fresh agent initially read only the run-experiment skill and baselines/phygile/docs/environment.md. No code, dependency, or configuration changes were made. SSH used BatchMode=yes, ConnectTimeout=12, and the authorized existing key/host. The commands below were executed verbatim in the documented work directory.

## Invocations and results

1. `cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017 && bash scripts/run_isaac.sh scripts/kernel_witness.py`
   - Exit status: 0.
   - Seed: 42. Tensor shape: 128 x 128. Loss: 254.7862548828125. Gradient finite: true.
   - Python 3.10.15; PyTorch 2.5.1+cu118; CUDA 11.8; NVIDIA RTX A6000.
   - Tracking package: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/upstream/source/whole_body_tracking/whole_body_tracking/__init__.py`.
2. `cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017 && bash scripts/smoke.sh`
   - Exit status: 0. Started 09:13:15.275 UTC; finished 09:13:42.781 UTC (about 27.5 seconds including SSH).
   - Observed device: cuda:0. Observed environments: 32.
   - Completed Learning iteration 0/1 using MoEPPO, with 768 total timesteps.
   - Collection: 4.763 seconds; learning: 0.341 seconds.
   - Mean value loss: 0.9102; mean surrogate loss: 0.1477; mean reward: -0.53.
   - Checkpoint confirmed by a separate read-only artifact listing: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_09-13-28_smoke/model_0.pt`, 67,520,342 bytes, modified at 09:13:41.623951739 UTC during this run. This witness confirmed file persistence, without deserializing the checkpoint.

## Document versus reality

- No command, working-directory, tracking-package, GPU, environment-count, or update-count divergence was observed. No package installation, override, or repair was required to run either command.
- GPU 0 initially contained 699 MiB of allocations and GPU 1 contained 8842 MiB. Existing jobs were left untouched, consistent with the document's shared-allocation instruction.
- The document does not describe the runtime diagnostics: headless display/GLFW and viewport warnings; a locked Kit key-value database; missing rendering-mode extension metadata and shader asset warnings; inaccessible OmniHub; git metadata lookup warnings; and `[gpu.foundation.plugin] DriverShaderCacheManager::init() called without a shutdown()!` on stderr. These diagnostics did not prevent the PPO update, checkpoint write, or zero exit status.
- The two witness outputs do not independently print an Isaac Lab version, so this pass does not separately establish the document's precise Isaac Lab 2.1.0 version assertion. The simulation did use the existing Isaac-Sim/4.5 installation paths.

## Saved evidence

Each executed remote command has an invocation file, full stdout and stderr logs, and an exit-code file in this directory, using the prefixes `preflight`, `kernel`, `smoke`, and `checkpoint_listing`. No detached process was used. The smoke finished within three minutes, so no long-running process inspection was necessary.

Conclusion: actual numerical kernel work, one PPO update, and checkpoint persistence completed successfully. This is a smoke validation only; it does not demonstrate converged policy quality or full research reproduction.
