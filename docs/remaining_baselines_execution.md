# Remaining official baselines, 2026-10-01

Server: ouyanghao@10.147.17.32. Root: /data/code/ouyanghao/oyh/conditional-human-motion-generation.
Warm reuse of configs/env-dreamcontrol-r3.json and the existing Isaac Sim 4.5 / Isaac Lab overlay. No package rebuild. Kick and Punch are complete and excluded.

Nine sequential tasks: Jump, Sit, Pick, Bimanual-Pick, Ground-Pick, Ground-Pick-Top, Button-Press, Open-Drawer, Open-Door. Each: original reference files, seed42, 8192 environments, 2000 iterations from zero, physical GPU1. Each successful training is followed by official 1000-environment, 500-frame evaluation and video validation. A failed task stops the queue; no silent resource reduction or retry from zero.

Independent preflight, run verbatim in PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 scripts/r3_dreamcontrol_remaining_queue.py --check'
```

Expected exit0 and `REMAINING_QUEUE_PREFLIGHT_PASS tasks=9 envs=8192 seed=42`. This validates launch inputs and the already confirmed live-state path, not a completed training or task score. PPO uses the previously validated optional nonblocking telemetry entry; its task, model, PPO, reward and data are unchanged. An independent renderer consumes actual sampled training poses on loopback UDP40123; R3_VIEWER_GPU=1 puts it on GPU1 to leave GPU0 available to CLoSD. Kit may retain a smaller graphics context on GPU0, so check actual device allocations before accepting CLoSD capacity. Neither renderer nor local viewer gates the running PPO process. Existing visibility confirmation is retained, not falsely described as a new user confirmation.

Launch only once after validation: `tmux new-session -d -s r3_dreamcontrol_remaining_20261001 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 -u scripts/r3_dreamcontrol_remaining_queue.py > artifacts/monitoring/r3_dreamcontrol_remaining_20261001_queue.log 2>&1'`.

Queue state: artifacts/monitoring/r3_dreamcontrol_remaining_20261001.state. Each task has independent logs/states and results JSON with checkpoint/video hashes. Models and licensed assets remain on the server. Read-only monitoring must confirm current iteration and log time, not only a running tmux session. Each task's final iteration1999, normal exit, model and positive video duration are required.
