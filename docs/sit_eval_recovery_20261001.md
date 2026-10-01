# Sit evaluation recovery, 2026-10-01

Warm unchanged `configs/env-dreamcontrol-r3.json` environment, Isaac Sim 4.5 and existing Isaac Lab overlay. No installation or model/reward/PPO edits. Sit completed iteration1999/2000 normally. Final model SHA256: `4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7`.

First official evaluation aborted with native `malloc(): invalid size (unsorted)` during simulation initialization, before loading the policy or reporting success rate. Host RAM and GPU memory remained available; this is not evidence of CUDA OOM. Specific root cause is unconfirmed. Keep the failed log and stopped queue states; never report zero success or restart Sit training.

Independent validation command, run verbatim from local PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 scripts/r3_dreamcontrol_sit_eval_recovery_20261001.py --check'
```

Expect exit0 and `SIT_EVAL_RECOVERY_PREFLIGHT_PASS final1999 hash_verified retry_only unchanged_budget`. This is input/lock validation, not successful simulation or a task score. Run no package installs or training during the independent check.

Only after validation, launch once using the original queue session name, which must no longer exist:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'tmux new-session -d -s r3_dreamcontrol_remaining_20261001 "cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 -u scripts/r3_dreamcontrol_sit_eval_recovery_20261001.py > artifacts/monitoring/r3_dreamcontrol_sit_eval_recovery_20261001_queue.log 2>&1"'
```

One clean process retries the exact official `play_eval.py` at 1000 environments, 500 frames, physical GPU1, video enabled. No physics configuration, seed, reference, checkpoint or budget changes. Official evaluation leaves the environment seed unset, which must be recorded separately from training seed42. The retry has a unique video/log id. It copies the final checkpoint, original logs and blocked states to `artifacts/recovery/Sit_20261001_final1999` and verifies the backup model hash. Original evidence is never overwritten.

Only a normal exit, reported success rate and positive video duration mark Sit complete. Then the unchanged original scheduler reacquires its single queue lock and skips completed Jump/Sit, proceeding to Pick and the other six pending tasks. On a repeated error it stops without auto-looping or retraining. Monitor `r3_dreamcontrol_sit_eval_recovery_20261001.state/.log` and queue log. Independent actual-state rendering remains separate; no monitoring process is stopped by this recovery.
