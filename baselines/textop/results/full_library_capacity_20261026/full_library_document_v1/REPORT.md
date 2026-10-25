# Full-library fresh document execution — independent report

2026-10-26, Asia/Shanghai. Result: **capacity and one bounded PPO iteration verified; training stability unresolved; formal training remains disabled.**

A fresh document-following agent executed `full_library_document_v1` exactly once, without repairing/retrying the experiment, reading implementation to repair it, changing packages/algorithms, saving weights, or starting formal training. The observed command, real exits, raw aggregate receipts and SDK logs are retained. The 67 independent evidence checks in `comparison.json` passed; these checks do not convert extreme finite losses into a stability acceptance.

## Inputs and exact invocation

Read the run-experiment skill and compute-environment contract, `FULL_LIBRARY_CAPACITY_ENVIRONMENT.md`, `configs/full-library-capacity-spec.json`, `SERVER_NOTES.md`, and local `results/full_library_capacity_20261026/continuation_v1.json`. Their input snapshots are retained here. No applicable ancestor AGENTS.md was found. The environment remained the declared warm tracker `b7a3bf0c`; the activated capacity-spec file SHA is `0f9dc72c4c616e673d343d5e8a49a091f48e40dc2cd9156d32a6e83854b19a40`.

The documented command was executed with only the required run-name substitution:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_full_library_capacity.py --run full_library_document_v1'
```

`invocation.txt` preserves this command, `invocation.stdout.txt` preserves its output, and `invocation_exit.json` records the actual local SSH/supervisor exit. The earlier independent SSH preflight failure reported by the parent was not rerun as a GPU experiment: this agent observed no existing document-run output or exit receipt before dispatch and made one GPU invocation.

## Preflight and actual exit

At 05:02:33 Beijing time, GPU1 had 40,726 MiB free, personal data storage had 8,754,601,984 bytes available, and host MemAvailable was 135,005,200 kB. No personal active SDK worker was present. Old MVAE57D PID3237858 remained `T (stopped)`; its launcher was sleeping. Historical local coordinator PID39924 no longer existed. The document output directory and exit receipt were absent.

The invocation ran from 05:03:13.672 to 05:06:30.256. Supervisor PID3778701 launched child PID3778706, whose SDK interpreter was PID3778713. The actual SSH/supervisor exit was **0**, the actual child exit was **0**, and `timed_out=false`; supervisor elapsed time was 195.53431447967887 seconds. The numerical receipt reports `TEXTOP_FULL_LIBRARY_8192_CAPACITY_PROBE_PASSED`.

At 05:07:07, all three document-run PIDs were absent, GPU1 was back to 40,726 MiB free, and old PID3237858 was still stopped. The live formal filter plan still had `formal_launch_enabled=false`. No process was stopped or resumed by this agent.

## Author/document comparison

| Evidence | Author | Fresh document |
|---|---:|---:|
| Actual child exit / timeout | 0 / false | 0 / false |
| Verified original sources | 8,278 | 8,278 |
| Original reference frames | 4,825,012 | 4,825,012 |
| Environments / rollout steps | 8,192 / 24 | 8,192 / 24 |
| Completed PPO iterations / actual Adam updates | 1 / 20 | 1 / 20 |
| Environment transitions | 196,608 | 196,608 |
| Policy / critic observations | 557 / 557 | 557 / 557 |
| Value-function loss | 3.3047911274572776e27 | 3.3047911274572776e27 |
| Surrogate loss | 3.3946707844734195e-08 | 3.3946707844734195e-08 |
| Entropy loss | 41.149227142333984 | 41.149227142333984 |
| Maximum parameter change | 0.17417466640472412 | 0.17417466640472412 |
| Peak Torch allocated bytes | 17,246,092,288 | 17,246,092,288 |
| Peak Torch reserved bytes | 18,358,468,608 | 18,358,468,608 |
| Sampled GPU1 total usage, MiB | 32,490 | 32,490 |
| Sampled GPU0 total usage, MiB | 12,741 | 12,741 |
| New checkpoints written | 0 | 0 |
| Raw completed attempts | 1,028 | 1,028 |
| Native close returned | false | false |
| Teardown fallback used | true | true |

Device samples include non-Torch allocations and other processes. The 0.65 allocator fraction is a Torch setting and does not cap every SDK/device allocation. The GPU0 observation cannot be attributed exclusively to this run. There is no claim of exact process-specific peak device memory from sampled totals.

Both receipts verify all 8,278 originals and their 16,556 stored original/mirror references, exact source/split/original-mirror binding, prepared/reference file SHA checks, and the same original-only training order. All five reference-manifest SHAs and all six prepared-manifest SHAs match. The shared reference-order fingerprint is:

`0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7`

The private arrays were read by the documented kernel, not copied by this reviewer. The comparison checks their aggregate receipt bindings; it is not an additional independent re-export or array-level reconstruction.

## Code, specification and backup identity

The author/document receipt fields match exactly, and post-run file-byte hashes agree with those fields:

| Item | SHA256 |
|---|---|
| Capacity probe | `7d25ad8238977fa6397080732c7ddade1c7414305ae2f254fbe42d5326357e90` |
| Full-library input verifier | `4b5784cf49e9b010de84be4194071cccb8f23a990fa179bb70b1157320265068` |
| Recording runtime | `ad3bb92f5836e5213450420e8f33deffd43443da7d451796adef3d6193ef6962` |
| Paper filter configuration | `e6b59d8fde8574d2c953279301dfc2df0c44ff88ac90683be995b7b97f0ff0d9` |
| Activated capacity specification | `0f9dc72c4c616e673d343d5e8a49a091f48e40dc2cd9156d32a6e83854b19a40` |
| Persistent backup acceptance receipt | `8c742cdbeef4773bfc66fea828fb68d9c91d70f656e9d9385a70c31241fe307e` |

The actual supervisor SHA is `4dcda5dfc9b81efc3a6719229505e1ad3cb731d52bd7beda836d455d22c52846`, matching the documented draft hash; its source was not modified. Tracker-spec file SHA is `57992eea2e2f6aa879493409a7bc2b4d320451e3f24090ed731f1d580a058a93`, and canonical parsed hash is `b7a3bf0c`. These are different hash inputs, not a mismatch. Each of the six collected author/document receipt, exit and SDK-log files also matches the corresponding remote file SHA.

## Extreme executed-state metrics and shutdown limitations

The raw SDK metrics reproduce the anomaly, even though the losses are finite and the weights changed:

| SDK metric | Author | Fresh document |
|---|---:|---:|
| `Metrics/reward/landing_force_mean` | 16,532,800,274,432 | 16,532,801,323,008 |
| `Metrics/reward/feet_slide_velocity_sum` | 139,968,976 | 139,968,976 |
| `Metrics/motion/error_body_pos` | 2.3834 | 2.3834 |
| `Metrics/motion/error_joint_vel` | 29.6085 | 29.6085 |
| Mean reward | -2.42 | -2.42 |
| Mean episode length | 23.55 | 23.55 |

The landing-force metric is not bit-identical across runs; both are extreme. Finite loss, 20 updates, and matching memory only establish that this bounded workload fits and executes. They do not establish physical plausibility, a stable value target, policy learning quality, or successful motion tracking. This review does not diagnose the cause or alter numerical math to suppress it.

Both runs disclose `native_close_returned=false`, `teardown_fallback_used=true`, `shutdown_grace_seconds=20`, `output_writers_closed=true`, and `shutdown_completion_verified=true`. The fallback occurred after the completed bounded run and closed writers. Actual process exit was independently observed; native SDK shutdown success is not claimed. Raw SDK warnings, including headless-window warnings and the driver-cache error, remain in the preserved logs.

## Document divergence and scope

The documented launch command worked without command repair or experiment retry. Its constrained capacity expectation is supported. Statements retaining pending status in historical document sections should now be superseded with this actual result, while the stated stability uncertainty must remain. Passing both bounded runs cannot authorize the formal 40,000-iteration run under the present task restrictions and unresolved anomalies.

Two read-only collector lookups were initially made against paths not specified as valid execution inputs: the continuation file was first sought under the remote root, and a guessed verifier filename was absent. The actual continuation was then read from the documented local results location, and filename discovery identified `scripts/full_library_inputs.py`; its actual hash matches both receipts. These collector misses are preserved in `preflight.txt` and `postflight.json`, and caused no GPU re-execution, source repair or data mutation.

All physical-filter, dataset-acceptance, paper-exact-scope, formal-filter-training, and generator-training flags remain false. The receipts continue to disclose 13 unavailable selected export sources and 403 unavailable author-private clips. Available-library capacity does not certify the unavailable/private paper corpus.

This evidence directory contains only aggregate receipts, stdout, logs, document/config snapshots, comparison metadata and review helpers. It contains no copied raw motion arrays, per-source indices, labels, USD assets, checkpoints or weights. Nothing was committed, pushed, or publicly uploaded by this agent.

Primary evidence: `document/receipt.json`, `document/exit.json`, `document/sdk.log`, `invocation_exit.json`, `invocation.stdout.txt`, `comparison.json`, `source_hashes.json`, `preflight.txt`, and `postflight.json`. Author counterparts are in `author/`. `artifact_hashes.json` records local evidence-file integrity.
