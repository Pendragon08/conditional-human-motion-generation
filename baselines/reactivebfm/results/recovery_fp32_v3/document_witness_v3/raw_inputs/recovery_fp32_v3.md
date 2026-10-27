# ReactiveBFM audited FP32 recovery v3

This is a supplementary numerical recovery, not a completed paper reproduction.
The Python/dependency environment remains unchanged (env-spec hash5d436660).
Original datasets, failed attempt, v2 trial, logs and checkpoints are retained.

Runtime: existing SSH ouyanghao@10.147.17.32, port22; local key
C:/Users/Pende/.ssh/id_ed25519. Python is
/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python.
Code directory is /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code_v3.

Restore model, EMA and AdamW from the verified periodic10020/10021 pair. The
launcher loads and inspects both files on CPU, pins its complete upstream Python
source tree and all recovery entrypoints, and GPU initialization verifies the
actual loaded optimizer count. Ambiguous trailing-checkpoint conventions are rejected.

The objective/data/global batch1024/LR1e-4/target1M/400k teacher-forcing warmup are
preserved. FP32 replaces BF16; FP64 is used only for norm reductions and clipping.
Pre-step, post-AdamW and EMA finiteness decisions are collective. Any rejected
attempt stops before EMA/schedule/save; an AdamW post-step failure records called
versus validated counters and stops without claiming rollback. Per-attempt heartbeats
are saved independently of successful-update multiples. Exact RNG/data-order replay
is not claimed. Resource cap: shared GPU0/1,65% of GPU memory per rank,CPU4/rank.

This is supervised planner training, not a newly started simulation-RL run.

## Independent eight-update invocation

Run verbatim once from local PowerShell, with no hidden setup or edits:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code_v3/scripts/launch_recovery.py --config /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/code_v3/configs/recovery_fp32_v3.json --output /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/document_witness_v3 --probe-updates 8"
```

After exit0, inspect status_rank0/1.json, health_rank0/1.json, supervisor_exit.json,
source_pins_verified.json and supervisor_stdout.txt in that output directory.
Both ranks must report PROBE_COMPLETED, optimizer_updates10029, initial10021,
resumed_updates8, attempts8, skipped_attempts0, optimizer_step_calls equal
validated_optimizer_updates10029, finite norms and ema_validated=true.
Supervisor exit is0, with model/model_avg and optimizer loading in stdout.
The final health.json has terminal process metadata; health.jsonl retains the
per-attempt trace. Any discrepancy fails the witness; do not repair or reuse the directory.

## Full continuation

Only after the witness passes, run the same launcher/config without probe-updates,
using fresh output /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/training_fp32_v3,
under screen reactivebfm_repair_fp32_v3. The first pair is saved after20 additional
validated steps (total10041), then the10k cadence continues. Configure a separate
backup-worker output/evaluation root. Retain raw ready-manifest bytes, launch/source
pins, status/health traces and supervisor logs. Record completion only after both
ranks finish successful updates and the final checkpoint is verified on GitHub.
