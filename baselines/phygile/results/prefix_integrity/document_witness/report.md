# Independent document-following witness

**Result: the bounded invocation passed. This is not a complete PhyGile reproduction.** A fresh witness executed the two documented lines exactly once on the authorized server, without implementation changes, dependency changes, or relaunches. Model identity was not exposed and is not inferred.

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
python3 scripts/validate_prefix_milestone.py
```

Execution ran on 2026-10-17 from 13:13:17 to 13:14:01 UTC. SSH naturally exited 0; `pipeline.json` says `COMPLETED`; stdout contains `PHYGILE_PREFIX_MILESTONE_COMPLETE`. GPU 0 had 2,968 MiB allocated beforehand; sharing its existing allocations was explicitly permitted by the invocation document.

| Stage | Natural exit | Seconds | Observed result |
|---|---:|---:|---|
| prepare | 0 | 0.165 | Guarded copy and three-file repair manifest created |
| regression | 0 | 1.919 | All 12 synthetic software checks passed |
| one_ppo | 0 | 21.512 | One iteration, 32 × 24 = 768 timesteps; checkpoint written |
| feedback_capture | 0 | 19.759 | Fixed 2000-iteration checkpoint replayed one training clip in 32 first episodes |
| feedback_gate | 0 | 0.265 | 0 accepted, 32 rejected at the unchanged 50-frame / 0.1 m gate |

The diagnostic checkpoint is `logs/rsl_rl/phygile_moe_pilot/2026-10-17_13-13-28_prefix_guard_diagnostic/model_0.pt` (67,348,310 bytes). CPU-only inspection found `iter=0`, 180 finite tensors, and 25 optimizer step tensors whose sole distinct value was 20. Its SHA-256 is `68439707ee5c166fe17a48fd141d1ed3e9b9264fe7d064f6c5ed70fc92bbe424`. No extra simulation or training was used for this inspection.

Saved feedback contains 869 finite 71D native states and 14-body positions. Every environment has contiguous frames from zero, 21–41 recorded frames, exactly one final failure flag, and no timeout. The installed IsaacLab `step()` updates physics buffers, calls `termination_manager.compute()`, and only afterward calls `_reset_idx()`. The capture wrapper copies tensors inside `compute()` before returning. This supports actual post-physics, pre-reset capture; separately paired post-reset tensors were not saved. Every gate record rejects incomplete continuation, failure termination, and MPJPE above tolerance. No threshold was adjusted.

Both simulation logs contain `PHYGILE_INDEPENDENT_REPAIRS_ACTIVE`; the PPO log identifies the guarded `train.py`. All three original and patched hashes match the repair manifest. The bootstrap prepends the guarded package path before tracking imports. **The original processes did not log individual module `__file__` values**: guarded activation and the entrypoint are directly evidenced, while per-module resolution is supported by the bootstrap and source integrity rather than a recorded runtime path assertion.

All 17 run artifacts (946,560 bytes) were copied into `evidence/document_run`; every local SHA-256 matches the remote file. Fifty-five upstream source paths have identical hashes in the prepare and final-stage receipts. The fixed parent checkpoint remains `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336`; the input motion remains `a4f3aa10ad1b17ad11508b5ea3b22664beb724ce7ea9148054d137cc8da6bd1b`. This witness did not independently hash every historical 500/2000-iteration output.

Document versus reality: all invocation success conditions were met. The remote `docs/environment.md` is stale: it says “Validation pending,” whereas the supplied local ledger records prior witnesses; both identify spec `fa5883c8`. Its general upstream import description also predates this invocation's explicit guarded-copy override. Logs skipped three unknown motion override keys and emitted nonfatal headless/display/shared-cache/shader-cache diagnostics. These did not prevent natural completion.

The run establishes bounded regression, one PPO iteration, and offline rejection of actual simulation feedback. It does not establish full curriculum training, generated motion quality, a 262D generator codec, online generation–execution–continuation, a standing terminal in simulation, closed-loop PPO adaptation, or real hardware performance.

Machine-readable findings: `report.json`. Full hashes and CPU/array inspection: `artifact_inspection.json`. Original process receipts, logs, and arrays: `evidence/document_run`. Transport evidence: `transport_receipt.json` and `invocation.log`.
