# User-requested quick reproduction:5000-round budget, full checkpoints retained

The user requested faster reproduction and asked to use fewer than40000 rounds.
The quick target is5000 completed iterations, followed by a small-sample real
tracking assessment. Results must be labeled QUICK_REPRODUCTION_NOT_PAPER_COMPLETE.
The original40k configuration and its acceptance rules remain unchanged; an early
snapshot cannot be represented as a completed paper policy or formal full-corpus
physical acceptance. Later continuation uses the private full checkpoint.

Warm reuse of the existing backend: IsaacSim4.5.0 / IsaacLab project2.1.0,
Python3.10.15 / Torch2.5.1+cu118 / RSL-RL2.3.1. No package or numerical-policy
changes. GPU1/Torch0.65/CPU0–3 remain the authorized resource shape. The current
policy continues running; no new SDK task may overlap it. Old57D/PhyGile/
ReactiveBFM and their checkpoints are protected.

## Fresh document preflight — read-only, exactly once

Run the following command from
C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation.
No repairs, retries, activation, training, process signals or config changes are
part of this witness. Do not add --deploy-disabled. This script uses the already
deployed disabled control files and checks all16 active runtime hashes.

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe baselines/textop/provider_tools/quick_budget_preflight.py --run document_preflight_v2
```

In PowerShell invoke the executable with the call operator `&`; preserve the
same executable, script and arguments. Expected actual exit0 and
QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED, signals_sent0, new_sdk_runs0. The current
teacher is below5000, so target_checkpoint_ready=false and enabled=false. Inspect
results/quick_reproduction_20261026/document_preflight_v2.json. Write the exact
command, actual exit/stdout/stderr and a doc-vs-reality report in a new folder
work/textop_reproduction_20261024/quick_budget_doc_witness_v2. Do not edit source.

CPU guard author already passed11 synthetic refusal/identity/backup cases with
zero signals/SDK. Author v1 remote preflight exited0 at2100 iterations, no target
checkpoint yet. Its exact command/result and independent review are retained.
V2 adds pidfd identity binding: three actual CPU-fixture checks passed on server,
including refusal after original-process exit;11 server guard checks also passed.
No signal was sent to a training process. A fresh v2 document preflight is required. These tests do not perform a real early stop or simulator work.

## Planned transition after enabling the quick coordinator

1. Observe the original trainer until its actual completed_5000.pt is stable and
   SHA-bound to its genuine receipt. Do not edit the active training plan/code.
2. Back up the full checkpoint privately onD and preserve it onserver. Publish
   only the existing inference-only export without per-motion sampling arrays;
   verify GitHub digest/size and both full/private hashes before any signal.
3. Only after successful backup, the new control helper identifies the exact
   owned trainer/session using its original launch proof, command/output path,
   UID and group. It sends SIGINT to that trainer for existing writer-finally
   cleanup; bounded TERM/KILL escalation uses the same original process handle.
   Signals use a pidfd tied to the original process plus its start-time identity.
   The old numeric-PID race noted in the v1 independent review was fixed before activation.
   It never uses pkill or touches another run/user. No signal occurs below the
   target, on a failed source run, or before backup verification.
4. Retain the original supervisor exit and interruption trace verbatim. The
   original run may report FAILED/KeyboardInterrupt and nonzero exit after the
   authorized interruption; a separate controlled-stop receipt records the user
   request and backed-up5000 checkpoint. Never relabel it40k FINISHED. A few
   transient updates after checkpoint creation may not be recoverable; existing
   file records are retained and recovery starts at the saved5000 snapshot.
5. After the original SDK exits/releases resources, use a separate quick/pilot
   evaluation profile accepting an explicitly early-stopped checkpoint. The
   existing v4/formal routes still require40000 and must not be weakened. Pilot
   real-tracking code/actual SDK checks are a next step, not completed here.
   Assess a small sample before any large-scale rejection; if the teacher is
   insufficient, continue from its checkpoint. Do not claim early-policy failure
   proves that the source motion is physically impossible.

The coordinator is intended to stop at the backed-up5000 node, not to reduce
VAE/LDM budgets silently. Generator data/step choices will remain explicit and
depend on the pilot quality. This changes the short-term priority to a quick
end-to-end demonstration while retaining the path to the original full baseline.
