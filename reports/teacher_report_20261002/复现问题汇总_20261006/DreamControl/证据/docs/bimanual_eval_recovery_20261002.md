# Bimanual-Pick evaluation recovery

Warm unchanged `configs/env-dreamcontrol-r3.json`, existing Isaac Sim4.5 and Isaac Lab overlay. Training reached1999/2000 normally; only the default video evaluation aborted during simulation initialization with `malloc(): invalid size (unsorted)`, exit1, before policy load. Do not count zero success, retrain or reduce environments. Model SHA256 `70d025af972fb3b1a3bb9534dbb938f838484b4233111045b53cc09061c9c1ab`, run `2026-10-02_00-21-24-Bimanual-Pick-a`.

Use one adapted official `play_eval.py --disable_fabric` evaluation: USD scene I/O instead of Fabric. Same1000 environments,500frames,video,physicalGPU1,checkpoint,physics,reward andreferences. Sit's preceding recovery demonstrated this supported runtime option; the specific native corruption root cause is still unconfirmed. Explicitly label the runtime adaptation, not identical-runtime replication. No package changes.

Fresh independent input/lock validation from PowerShell, verbatim:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 scripts/r3_dreamcontrol_completed_eval_recovery.py --task Bimanual-Pick --model-sha 70d025af972fb3b1a3bb9534dbb938f838484b4233111045b53cc09061c9c1ab --check'
```

Expect exit0 and `COMPLETED_EVAL_RECOVERY_PREFLIGHT_PASS task=Bimanual-Pick final1999 hash_verified envs1000 frames500`. This validates final model, failed log, task state, confirmed display gate, official flag availability, unused adapted log and single queue lock; not successful simulation. Independent checker performs no install/edit/train/eval.

After validation launch once, only when original queue session is absent:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'tmux new-session -d -s r3_dreamcontrol_remaining_20261001 "cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 -u scripts/r3_dreamcontrol_completed_eval_recovery.py --task Bimanual-Pick --model-sha 70d025af972fb3b1a3bb9534dbb938f838484b4233111045b53cc09061c9c1ab > artifacts/monitoring/r3_Bimanual-Pick_usd_eval_recovery_20261002_queue.log 2>&1"'
```

Backup the final model, original logs and blocked states under `artifacts/recovery/Bimanual-Pick_20261002_final1999`, with hashes. Never overwrite original failed logs. State/log prefix `artifacts/monitoring/r3_Bimanual-Pick_usd_eval_recovery_20261002`. Success requires normal exit, official metric and nonempty positive-duration video; only then update task completion and return to the unchanged original single-lock scheduler, skipping completed tasks and starting Ground-Pick. Repeated failure stops; no automatic loop. Training seed42 differs from official evaluation's unset environment seed. Independent actual-state monitor remains separate and cannot block computation.
