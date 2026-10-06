# Official cabinet asset recovery, 2026-10-03

The Open-Drawer attempt `2026-10-02_16-33-03-Open-Drawer-a` stopped during `gym.make`/scene initialization. It produced no PPO iteration or checkpoint. The scheduler correctly marked BLOCKED even though the outer simulator returned exit0. This is not a zero-success evaluation.

`Training/assets/cabinet.usd` was a 130-byte Git LFS pointer. Its declared official object is 17599 bytes, SHA256 `30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c`. The official binary was staged and verified from the pinned DreamControl commit `ec28b508313d6837de52c68247fbc8cf81fe4935`:

https://media.githubusercontent.com/media/GenRobo/DreamControl/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/assets/cabinet.usd

Only this official LFS asset is hydrated. Task configuration, articulation settings, reward, PPO, motion references and formal budgets are unchanged. Open-Door uses the same asset, but a successful Open-Drawer diagnostic alone does not certify Open-Door completion. Original pointer, failed log, BLOCKED states and empty failed run remain archived under `artifacts/recovery/Open-Drawer_asset_20261003/`.

## Existing environment and independent diagnostic

Warm reuse: `configs/env-dreamcontrol-r3.json`, existing Isaac Sim 4.5.0 bundled Python and existing Isaac Lab source overlay. No package installation. Provider ledger: `docs/dreamcontrol_ssh_environment_ledger.md`. The user has already confirmed the actual-state monitoring channel; UDP40123/independent renderer cannot block PPO. CLoSD has a separate pending confirmation and is not started by this procedure.

A fresh witness agent must follow this exact PowerShell command, without changing flags, installing packages, deleting logs or starting a second queue:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; python3 -u scripts/r3_cabinet_asset_witness_20261003.py'
```

Expected: exit0, `CABINET_ASSET_WITNESS_PASS`, official Open-Drawer actual physics initialization and exactly one PPO update at seed42, 16 environments, GPU1. A diagnostic `model_0.pt` is explicitly excluded from formal results. The script checks the official asset hash, absence of an active trainer/queue and refuses an existing witness log. Report a mismatch verbatim; do not repair or retry independently.

Only after this independent witness passes, recheck that the failed formal log has no `Learning iteration` and failed run has no checkpoints; preserve/move the log into recovery, then restart the original unique scheduler. It skips all completed tasks and resumes Open-Drawer/Open-Door at the unchanged 8192/2000/seed42 budgets. A new task must write actual PPO iteration before being called running. If initialization fails again, retain the evidence and stop; no unlimited retry or downsizing.
