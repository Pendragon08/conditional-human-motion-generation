# Strict SDK replay diagnostic, warm tracker b7a3bf0c

Read SERVER_NOTES.md and configs/strict-replay-spec.json. No dependency or
shared-source changes. GPU1,CPU4,Torch65%,headless. The explicitly SHA-pinned
checkpoint completed only2PPOiterations; this is infrastructure validation,
not a trained tracking result or accepted data. Two original references have
85and90frames. Formal evaluation remains disabled pending actual40kpolicy.

Fresh independent command, exactly once unchanged in PowerShell:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_strict_replay.py --run strict_replay_document_v2 --mode witness'
```

Require actualexit0, no timeout, TEXTOP_STRICT_REPLAY_SUPERVISED_EXIT0 and
TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED. Receipt must show reference_sources2,
num_envs2,finished_trials2,optimizer_updates0,checkpoint_completed_iterations2,
diagnostic_dynamic_rollouts_executedtrue,physical_filter_executedfalse,
dataset_acceptedfalse,full_generator_training_startedfalse and closed writers.
Require reset_reference_alignment_refreshed_without_time_advancetrue; retain
complete/failed counts and completion_branch_exercised as observed. All failures
are permissible for this untrained diagnostic. A separate diagnostic_prefix_
probe uses explicitly shortened2frame control-flow fixtures, one real SDK step
each, to exercise natural-end signal handling. Require its
sdk_motion_end_branch_verifiedtrue and full_original_motion_passes0. Those
fixtures NEVER add passing full motions or acceptance. The original85/90frame
reference arrays/trial results stay unchanged. A false
completion_branch_exercised still means no FULL original completed; the2frame
fixture verifies SDK end handling only. Expected latest author3:17vectorized
calls/68scenephysicssteps,2failed full trials,0complete full trials, plus2
complete prefix fixtures. Verify actual values and report differences.

Standard pelvisheight0.25m/EEheight0.5m/orientation term restored. Deterministic
frame0 reset is an explicitly declared evaluation choice. Default10stimeout is
removed and replaced by exactT-1transition motion-end accounting; first actual
failure ends that trial and subsequent auto-reset episodes cannot extend it.

Author_v1 exited0 but both motions terminated at the first control call. Review
found pinned MotionCommand's aligned-reference buffers stay zero/stale until
its first post-termination command update. This contaminates first-step height
checks. Historical source and logs retained. Author_v2 refreshes exactly the
released reference-alignment transform after reset without advancing time,
changing EMA, relaxing thresholds or changing physical states. Failure term
counts and initial body discrepancy are recorded. The policy is unchanged.

Public evidence paths on server: tracker_validation/strict_replay_document_v2/
receipt.json and native_teardown_stack.txt; receipts/strict_replay_strict_replay_
document_v2_exit.json; logs/strict_replay_strict_replay_document_v2.log. Copy only
these public artifacts. Do not copy checkpoints, per-motion private JSON or
reference arrays, inspect/repair source, retry, stop other tasks or publish.
Verify source/spec hashes without source inspection. Save command/stdout and
doc-vs-reality report in work/textop_reproduction_20261024/strict_replay_doc_
witness_v2. Nativeclose remains blocked; after saved output and closed writers,
20sfallback can exit only this completed child. Distinguish actualexit0 from
native_close_returnedfalse/teardown_fallback_usedtrue.
