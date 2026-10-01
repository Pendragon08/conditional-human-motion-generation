# Sit official evaluation with USD I/O

Warm existing Isaac Sim4.5/Isaac Lab overlay from `configs/env-dreamcontrol-r3.json`; no package edits. Training completed1999; model SHA256 `4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7`. Two original video-enabled evaluations aborted during initialization with native malloc errors. One 120-second camera-disabled diagnostic reached policy loading, then intentionally stopped. It supplies no official score or video.

This bounded adapted evaluation uses official `play_eval.py --disable_fabric`: USD I/O replaces Fabric scene I/O. Keep original Sit scene, references, PPO checkpoint, physics settings, 1000 environments and500frames. Video remains enabled. This is an explicit runtime adaptation, not an identical-runtime replication. Specific cause of native heap corruption remains unconfirmed; do not claim Fabric is proven to be the root cause.

Independent input-only validation, run verbatim in PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 scripts/r3_sit_usd_video_eval_20261001.py --check'
```

Expect exit0 and `SIT_USD_VIDEO_EVAL_PREFLIGHT_PASS same_model_envs1000_frames500 official_disable_fabric`. Do not run installs, edits or evaluation in this independent pass. Model hash, completed no-camera diagnostic, failed original attempt, backup, queue lock and unused adapted log are checked. This is not a successful simulation claim.

After check, launch once in original absent session:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'tmux new-session -d -s r3_dreamcontrol_remaining_20261001 "cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 -u scripts/r3_sit_usd_video_eval_20261001.py > artifacts/monitoring/r3_sit_usd_video_eval_20261001_queue.log 2>&1"'
```

Log/state prefix `artifacts/monitoring/r3_sit_usd_video_eval_20261001`; previous failed logs and blocked snapshots are preserved. Normal exit plus success rate and positive video duration are required. Only success marks Sit complete with explicit adaptation and returns to the unchanged original single-lock queue, which skips completed Jump/Sit and starts Pick. Failure stops with no automatic repeat. Independent monitor stays separate. Official eval environment seed is unset, unlike training seed42.
