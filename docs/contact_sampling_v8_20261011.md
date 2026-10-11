# V8: complete native simulate before reset restores its cache

This is a measurement diagnostic, not a method innovation. V6 invoked the wrong
collector; V7 fenced at the end of reset and exposed inconsistent rigid-body
cache restoration. Preserve both attempts and their cost. V8 uses a forwarding
Gym proxy to wait immediately after each native simulate, including the reset
warmup, before the native reset's second tensor restoration. It neither adds
simulation steps nor refreshes. Generator/controller/conditions remain frozen.

Read the unchanged runtime ledger at .aris/compute/closd-baseline-20261010.md.
Fresh document witness: execute the exact command below once, no retries, edits
or --run. Check exit code, sentinels, and metadata rather than sentinel alone.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v8_20261011.py --check
```

Expected exit0 and CONTACT_SAMPLING_BATCH_PASS repeat_a plus
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS. Inspect
results/contact_sampling_v8_20261011/raw/repeat_a: 12 original attempts retained,
120 valid frames if completed; initial_reset_gate_pass; seven exported initial
state differences and four rigid-cache differences all zero; recorded effective
SimParams, completion_fence_calls at least120, adapter source hash matching V8.
A task failure is retained and does not by itself invalidate documentation;
measurement assertions and missing instrumentation invalidate the witness.
After witness, root runs --run for the independent repeat_b. Repeatability and
physical-metric validity are not certified by this first-batch witness.

Old thresholds and targets unchanged. All failures counted toward 72GPUh total,
4GPUh measurement; data free space at least20GiB, new output total at most15GiB.
No installs, file deletion, paid APIs, real robot tests, or Overleaf result edits.
V8 binds collector, queue, predecessor adapter, helpers, evaluator, protocol and
configuration before execution. Incomplete paths may not silently be retried.
