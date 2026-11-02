# TextOp tracker backend validation, revision2

Ledger: SERVER_NOTES.md. Canonical spec: configs/tracker-env-spec.json,
hash b7a3bf0c. Warm read-only Isaac Sim4.5.0 / Isaac Lab project2.1.0
(distribution0.36.21), Python3.10.15, Torch2.5.1+cu118, NumPy1.26.0,
Gymnasium0.29.1, RSL-RL2.3.1. No shared packages/assets changed.
Headless/shared GPU reproduction is explicitly authorized. Physics/MLP GPU1,
CPU affinity/library threads4, Torch cap0.65.

This diagnostic uses8 G1 instances, real32 physics steps at200Hz,29 joints,
name-mapped FK, and a557→2048→1024→512→29 seeded MLP backward pass.
It has no ground, contact test, optimization or physical filtering. Body count30
is expected after fixed URDF joints are merged.

## Shutdown issue and explicit workaround

Native SimulationApp.close did not return in first author/document runs.
V2 timed out after240s. V3 was rejected even though childexit0 because its receipt
recorded an optional Replicator import failure. All failure evidence is retained.

The current diagnostic uses SimulationContext.forward for FK, stops physics and
clears callbacks before close. If native close still hangs, a20s watchdog ends
only this completed diagnostic process after output writers are closed and its
receipt is saved. It records teardown_fallback_usedtrue and
native_close_returnedfalse. This is not native SDK shutdown success. The
supervisor retains the actual child exit/timeout and numerical status.
No camera/video writer or unsaved optimizer exists in this diagnostic.

Author_v5 passed, then used the explicit fallback: actual childexit0,
timeoutfalse, elapsed32.05s. Independent execution must pass too.

## Execute exactly once in PowerShell

Output document_v2 must not exist. No repairs, retries, flag changes, stopping
other jobs or training launches are permitted.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_tracker_kernel.py --run document_v2'
```

Expected direct commandexit0 and TEXTOP_TRACKER_SUPERVISED_EXIT0. Inspect:

- /data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/tracker_kernel_document_v2_exit.json
- /data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/document_v2/receipt.json

Supervisor child_exit_code0, timed_outfalse. Numerical receipt:
TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED, dof29, num_envs8, physics_steps32,
optimizer_updates0, physical_filter_executedfalse, full_filter_training_readyfalse,
ground_presentfalse, tracking_success_evaluatedfalse, output_writers_closedtrue,
spec_canonical_hashb7a3bf0c. If fallback is used,
shutdown_completion_verified must be true. Report native-close flags as observed.

Witness source SHA256:
a224ffba40f9189e882121397b9f4702bc43f6973b6f1f064952d4bf69eec1f2.
Shutdown helper SHA256:
f42904e6aa8517156c64505155ff1cabd62ebeaf2bda10ed3ff67c803ed317c9.
Shared URDF SHA256:
dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b.
Exact TextOp author asset hash is unpublished; equivalence is unconfirmed.

Private reference arrays/generated USDs stay on server. Only receipts/logs may
be published. Passing this backend test does not validate the full Tracker MDP,
8192-env capacity,40k filter policy, final corpus or generator training readiness.
