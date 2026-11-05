# Bounded online-loop and exact interaction-budget resume witness

This is a new application integration on an unchanged existing environment.
Do not install packages or modify TextOp. This does not launch formal8912-env
training, establish task success, or reuse PPO updates under a different name.

Runtime: existing Isaac4.5 Python/Torch2.5 from
`stage3_pick29_20261103/env_spec_v2.json`. The wrapper loads exact environment,
working directory and interpreter from the immutable bridge spec c7724ac5.
TextOp native primitives remain pinned to commit0525a23a2dcfa61f8543f98b84c075fe66d9d286.
The current PhysX robot, hands, object, real reward/reset and live102-input Actor
are retained. World-model state112, action31; private reward context109 never
enters Actor. Algorithm-specific learned reward/termination heads and binary
hand straight-through gradient are explicitly documented adaptations.

Execute this command verbatim once from the local workspace:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 10.147.17.32 'python3 /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103/src/stage3_supermimic_online_witness.py --name online_doc_v1 --num-envs 4'
```

The system `python` is Python2; explicitly use `python3` for the wrapper. It
launches the existing Isaac Python internally; no dependency rebuild is needed.
The wrapper uses exclusive directories, two child launches each bounded240s,
and saves original child exit codes. Do not rerun if a child is still running or
if the output directory already exists; preserve failed attempts.

Expected result:

- First process performs72 actual vector steps, including64 world-model warmup
  sampling steps,4 world-model-only updates and3 imagined-policy updates.
- Second process restores model/optimizers/norms/RNGs/replay/UTD/real-step budget
  at72 and ends at96 actual vector steps. Physics reinitializes on resume;
  bitwise trajectory continuation is not claimed.
- Total384 actual transitions for4env,16 world-model updates,12 policy updates,
  remainingUTD tenths8; no doubled warmup or resetting the interaction budget.
- Full checkpoint/replay roundtrips pass; Actor receives only102 live inputs.
- Original exit0 and sentinel `SUPERMIMIC_ONLINE_RESUME_PASSED`.
- Proofs under server `stage3_supermimic29_20261103/reports/online_doc_v1/`:
  `report.json`, `prefix/status.json`, `resumed/status.json`, child/exit logs and
  both immutable full resume checkpoints. Failures are retained in place.

Author variant `online_author_v1` already passed8env/768 actual transitions.
This independent documented variant intentionally uses4env/384 transitions.
After the witness, formal preparation still needs8912-env optimizer/replay
capacity profiling, final evaluator and guarded launch with11GiB startup,
8GiB+356MiB running floor,12GiB ownGPU/32GiB RSS limits. Nothing in this
bounded witness admits or starts a formal job.
