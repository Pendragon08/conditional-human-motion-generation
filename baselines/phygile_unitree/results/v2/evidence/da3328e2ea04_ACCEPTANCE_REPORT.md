# Fresh document witness: FAILED

The exact documented code_v10 / closure_unitree_v2 command was executed once. It returned submit exit 0 and PID 3158958 at 2026-10-22 12:00:48 UTC, but the worker died before startup:

```
taskset: failed to set pid 3158958's affinity: Invalid argument
```

The first read-only poll found no process, no run output files, and no verification.json. Missing completion is failure. The document claim diverges from reality at the CPU affinity launch: a successful submit receipt did not mean the worker started. No repairs or retries were performed.

Acceptance status: FAILED_DOCUMENTED_INVOCATION. Actual Unitree physics, replay, two accepted proxy feedback windows, and actual diffusion rejection were not reached. The generator remains untrained/unconditional; these criteria could support engineering validation only and would not demonstrate stable semantic motion.

Evidence includes the exact invocation, complete submit tool output and receipt, stdout/stderr, read-only poll/process snapshot, the code_v10 source hash manifest captured after failure, and before/after document/submit hashes. The local document and submit hashes match. This witness did not capture a remote source hash manifest before launch, and therefore does not claim an independent before/after remote source comparison. No original datasets/checkpoints or ReactiveBFM resources were modified; no other jobs were killed. No raw arrays were transferred.
