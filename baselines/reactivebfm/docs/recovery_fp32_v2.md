# ReactiveBFM FP32 recovery, 2026-10-22

This is a supplementary recovery variant, not evidence of the paper's performance.
The original failed attempt, its code/configuration and checkpoints are retained.
The Python/dependency overlay is unchanged; no packages are installed or upgraded.

## Runtime and changes

- Existing SSH server: `ouyanghao@10.147.17.32`, port22.
- Local SSH key: `C:/Users/Pende/.ssh/id_ed25519`; use BatchMode, never copy credentials.
- Python: `/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python`.
- Repaired code: `/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code`.
- Config: `configs/recovery_fp32_v2.json` under that code directory.
- Reuses the read-only original `code/upstream` through a symlink. Original `train_full.py` is unchanged.
- Restore model, EMA and all AdamW moments/counters from the ready 10,001-update pair.
- Use FP32 forward/backward instead of BF16; FP64 reductions only for norms/clipping.
- Keep dataset, objective, global batch1024, learning rate1e-4, 1M target and400k teacher-forcing warmup.
- All ranks agree before applying a step. EMA updates only after an applied step.
- Independent per-attempt health receipts expose applied/skipped updates. One rejected
  attempt terminates the recovery with evidence, before the schedule counter advances.
- GPU allocation cap65% per rank, CPU4/rank. Shared GPUs were explicitly authorized.
- This restores optimizer state, not exact random-number or data-iterator replay.
- This is planner supervised training, not a new simulation-RL run. Independent
  MuJoCo checkpoint diagnostics must not be called a live RL-training viewer.

## Independent document-following witness

Run this command verbatim from the local PowerShell environment; do not add hidden setup.
It creates one fresh, bounded eight-update recovery. It must fail if the destination exists.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code/scripts/launch_recovery.py --config /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code/configs/recovery_fp32_v2.json --output /home/ouyanghao/reactivebfm-full-20261021/recovery_doc_witness_v2 --probe-updates 8"
```

After exit0, read `status_rank0.json`, `status_rank1.json`, `health_rank0.json`
and `health_rank1.json` in the output directory. Both ranks must show
PROBE_COMPLETED, actual optimizer_updates10009, resumed_updates8, no rejected
attempts, finite norms and matching initial optimizer_updates10001. Inspect
supervisor_stdout.txt for loading both model and model_avg and optimizer state.
Any discrepancy is a failed witness; do not repair or rerun into that directory.

## Full continuation

After the witness passes, use the same launcher and configuration without
`--probe-updates`, with a new output directory
`/home/ouyanghao/reactivebfm-full-20261021/repaired_fp32_v2`. Start under a
dedicated screen session, preserving supervisor stdout/stderr. The first recovered
checkpoint is saved after20 additional applied updates (total10021), then the
original10k checkpoint cadence continues. Rebind the backup worker to this new run,
local output directory and Release; preserve raw ready manifests and their hashes.

Never overwrite the original attempt or delete datasets. Old checkpoint copies
may only be considered by the existing exact-size/SHA-verified backup policy;
the newest checkpoint is always retained. Full continuation remains unfinished
until both ranks have the required successful updates and the final pair is backed up.
