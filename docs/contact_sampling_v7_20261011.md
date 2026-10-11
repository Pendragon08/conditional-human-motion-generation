# V7: explicit simulation-completion fence

This is a measurement diagnosis, not a candidate motion-generation innovation.
The same 4 validation sources, 3 offsets and generation seed 11 are repeated
twice. Keep original targets, controller, generator, thresholds and old results.
No new physics steps or refreshes are inserted. Wait for native PhysX completion
after the native physics step and native reset, before tensor readback. Retain
the V5 CUDA synchronization and measure any contact-tensor mutation by reward.
The hypothesis is that an explicit completion fence improves force repeatability;
it is not presumed true. An unchanged classification at a zero pass rate does
not validate the physical metric.

## Independent document-following witness

Run the exact command below once. Do not retry, edit source, or run --run.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v7_20261011.py --check
```

Expected: exit 0, CONTACT_SAMPLING_BATCH_PASS repeat_a and
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS. Read metadata/receipt/numerical certificate
in results/contact_sampling_v7_20261011/raw/repeat_a. Check 12 retained first
attempts, exported reset gate and seven first-action state differences of zero,
recorded completion fences (at least 120 for a completed 120-frame collection),
and recorded simulator parameters. This witness does not require task success
or establish repeatability. The executor subsequently runs --run for repeat_b.

Environment specification is unchanged and warm-reused. No package changes,
VLM calls, robot hardware, file deletion or Overleaf experiment updates.
All old and new failures count toward the 72 GPU-hour cap; this test counts
toward the 4 GPU-hour measurement cap. Data free space must be at least 20GiB,
total new output at most 15GiB. Bind collectors, queue, helper/evaluator,
configuration, protocol and source manifest before execution. Preserve failure
directories; never silently rerun an incomplete batch.

V6 queue incorrectly invoked the V5 collector. Preserve its outputs and cost;
V7 explicitly invokes contact_sampling_collect_v6.py and asserts recorded fences.
This repairs invocation only, not the measurement threshold or method.
