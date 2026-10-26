# Read-only quick-budget controller safety review

Scope: static review of scripts/quick_budget_control.py, scripts/quick_budget_guards.py, and provider_tools/quick_budget_coordinator.py only. No additional server call, test, SDK, signal, activation, or source edit was performed. This report is not evidence that an actual controlled stop has succeeded.

Verdict: the normal documented path enforces the 5000-iteration and backup gates, preserves the original supervisor exit, and fails closed on transport/backup errors. One process-identity race remains relevant to the strict requirement that no other run can receive a signal; resolve it before activation if that requirement is absolute.

## Signal-identity gap: initial SIGINT uses a stale numeric PID

quick_budget_control.py:79 obtains trainer/group identity from the earlier /proc snapshot (scan at67-78). It then checks readiness, reads and hashes the checkpoint, validates the backup receipt, and writes the pending receipt. At117 it calls os.kill(trainer, SIGINT) using only the saved numeric PID. There is no pidfd binding or process-start-time identity check at the signal operation.

Concrete failure sequence: validation identifies the original owned trainer; that trainer exits while the intervening work runs; its PID is reused by a different process of UID1013; os.kill at117 targets the new process. Linux UID permissions protect a different user's process, but they do not distinguish two runs owned by the same UID. The later TERM/KILL branches recheck group/UID (and session for TERM), so this finding specifically includes the first SIGINT, which has no such recheck. This is a static race, not an observed incident or a rerun test result.

Suggested resolution for strict identity safety: acquire and retain a pidfd while verifying the target's identity and send SIGINT through that handle. A final /proc recheck narrows the race but does not atomically remove PID-reuse ambiguity. Keep the owned-group restrictions for any escalation.

## Target and backup gates

- Guards:9-14 requires the declared5000/40000 targets, genuine RUNNING_FORMAL_FILTER_TRAINING/formal/8192 state, the frozen original training-plan SHA, and optimizer_updates=completed_iterations*20. Reached means at least5000.
- Control:80-87 requires the completed_5000.pt file older than30s and binds its exact SHA/size to latest_checkpoint with5000 completed iterations and100000 optimizer updates.
- Control:97-108 requires early_stop_enabled=true, ready=true, no pre-existing exit, and validate_backup success before creating pending and reaching any signal.
- Guards:30-40 binds the backup receipt to exactly5000/100000 and the same original server checkpoint path/SHA/size; requires the private two-location and inference three-location verification flags, original server retention, no deletion, and no public upload of the full checkpoint.
- The backup producer and stored objects were outside this three-file review. These consumers verify the receipt and its checkpoint binding; this review alone cannot independently establish that every producer claim is true.

The exact latest_checkpoint match is conservative: if the active receipt later advances to a newer saved checkpoint, finish refuses before signaling even if the5000 file remains valid. The parent supplied the frozen schedule: save at1000 and5000,10000,...40000, never5100. At the supplied measured6.36s/iteration, a backup bounded to1200s does not plausibly reach10000 from5000. Therefore this is an intentional late/missed-target limitation, not a current blocker. My earlier5100 example was incorrect and is withdrawn.

## Other-run ownership and truthful completion

The target ROOT/RUN/TRAINER/OUT are fixed. validate_group checks the original launch command/output/mode, PGID>1, every enumerated member's UID1013/group/session, and exactly one Python trainer with the exact trainer/output arguments. No pkill, broad user-wide selection, or unrelated output path is present. Aside from the numeric-PID race above, no normal-path route to target another run was found in these files.

Control:36-48 does not overwrite either original receipt. It embeds the original supervisor exit, hashes its unchanged file and final receipt, records USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE, sets full_paper_training_completed=false and quick_checkpoint_completed_iterations=5000, and discloses unrecoverable transient updates. The coordinator uses QUICK_REPRODUCTION_NOT_PAPER_COMPLETE and a separate pilot-ready state; it dispatches no simulator or pilot SDK. An interruption is not relabeled as40000 completion.

## SSH/backup/error behavior

- Observe SSH errors, invalid JSON, source/plan mismatches, failed backup, backup timeout, bad backup status/hash, or failed server-proof transfer raise into NEEDS_INSPECTION. There is no automatic retry or fallback to signaling. Before finish these paths leave the trainer running.
- If the backup proof already exists with different text, the exclusive-write path refuses rather than replacing it. A local cancellation check occurs again after backup, and remote finish independently checks its enabled plan and gates.
- If SSH/control fails during finish, a signal may already have been delivered; the coordinator correctly does not assume otherwise or resend. The server pending receipt is created before the first signal. A later invocation seeing pending never sends a duplicate signal: it waits for the original supervisor exit, then validates backup and completes bookkeeping.
- A missing exit after the bounded INT/TERM/KILL windows raises and retains pending evidence. If trainer identity disappears before escalation, the code does not fall back to a broader process search.
- A publish failure after stopping marks local state NEEDS_INSPECTION; the server controlled-stop receipt remains separate and preserves the actual original exit. No new SDK is dispatched as error recovery.

The guards are implemented with Python assert statements, so their safety contract assumes the documented non-optimized interpreter execution (no -O/PYTHONOPTIMIZE). This source-only review did not inspect or change the server interpreter environment.

## Reviewed byte hashes

- scripts/quick_budget_control.py: a6b912c29e7741131abbbeebf284153dcbab6aba9f668f0ab0813ce7b2f4e981
- scripts/quick_budget_guards.py: 866a2ac63440cb5c220b6b2007d37e39624113735fe2c067c1c384ccc7eb7367
- provider_tools/quick_budget_coordinator.py: 519fdf81ff3d76ae4ad71737fae9c517aefeccc6dcf1faec8b5a5f120e52dd64
