# Full-schedule runtime witness (frozen invocation v1)

The user authorized full training and a simulation closed loop on the original
dual RTX A6000 server. Existing jobs must continue. The environment is the
unchanged, previously witnessed overlay declared in configs/env_spec.json.
The new execution policy is configs/full_training_v1.json: two ranks, batch512
per rank/global1024, EMA, official BF16 option and supplementary non-reentrant
DiT block activation recomputation. Full target1M updates, warmup400k.

This witness runs sequentially: nonzero-gradient FP32 parity; 4 real DDP updates
using all three released datasets; an 8s MuJoCo static-reference tracker check;
and 100 real closed-loop control steps using the old101-update pilot model.
It does not launch the full job or write large checkpoints. The pilot model is
expected to fall: execution is distinct from motion/control quality.

Execute the following single SSH-side command exactly once, without additional
setup. Its output directory must be absent. No package, source, config, document,
raw data or other process may be changed. Do not launch concurrent GPU work for
this project. The original user jobs remain running.

```sh
/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code/scripts/full_runtime_witness.py --runtime-root /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021 --output /home/ouyanghao/reactivebfm-full-20261021/full_witness_document_v1
```

Success requires exit0 and receipt status FULL_RUNTIME_WITNESS_PASSED, both ranks
4 actual updates, global batch1024, FP32 gradient parity tolerances1e-6/1e-5 with
at least200 nonzero gradient tensors, finite/upright static tracker hold, and
100 closed-loop policy calls/400 actual MuJoCo integration steps with >=2 planner
calls. No claim of a trained planner, real-time execution or paper metrics follows.

SSH host and identity are in the existing provider ledger. Preserve exact command,
stdout/stderr, receipt, source/config/doc before/after hashes, start/end UTC and
remote receipts locally under results/full_training_20261021/document_witness_v1/.
Copy only generated JSON/text evidence, not third-party motions or checkpoint bytes.
Report all differences directly; do not repair anything silently.
