# Independent solid-floor PPO document witness v2

Date: 2026-10-26 Asia/Shanghai. Run: `solid_floor_ppo_document_v2`.

The exact documented command was executed once and completed with actual SSH, supervisor and child exit code 0, no timeout, in 284.192919 seconds. The bounded result matches the documented ten PPO iterations and 200 optimizer updates. There was no rerun of the author or existing document runs. No formal training, physical filtering, data acceptance or generator launch occurred.

This is a fresh agent-following-document execution under the run-experiment skill. I read the skill, compute-environment contract, `SOLID_FLOOR_PPO_ENVIRONMENT.md`, the existing `SERVER_NOTES.md` ledger, the declared specification and relevant frozen source/configuration files. No applicable ancestor AGENTS.md was found. Local byte snapshots are retained under `input_snapshots/`.

## Invocation and independent preflight

Executed verbatim, exactly once:

```text
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_solid_floor_ppo.py --run solid_floor_ppo_document_v2 --case solid_floor"
```

All 42 preflight checks passed before dispatch. All 18 checked local/server source and configuration files had equal actual byte SHA256 values; all eight declared source/specification bindings and both private-input bindings matched. The warm canonical environment hash was `b7a3bf0c`. The author exit proof existed with child exit 0; the new run directory, exit proof and log were absent. GPU1 had 40,726 MiB free, available host RAM was 138,242,904,064 bytes, and free data storage was 8,508,723,200 bytes, above declared thresholds.

An initial deliberately broad read-only process check also noticed the old `launch_training.py` supervisor. Its preserved evidence is `preflight.initial_broad_process_check.json`; this did not execute the documented command or dispatch a GPU job. Independent `/proc` inspection then established PID3237857 was sleeping in `do_wait`, waiting on its old 23DoF/57D `train_official.py` child PID3237858, which was stopped. This known held pair is explicitly preserved by the documented supervisor. Neither process was stopped, resumed, killed or changed. The final preflight excluded only that independently verified passive pair and found no active personal SDK worker.

Observed server supervisor PID3808407, wrapper child PID3808412 and SDK worker PID3808419. The wrapper and actual worker had CPU affinity `[0,1,2,3]`; the documented physics/Torch target was `cuda:1`, four Torch threads and allocator fraction 0.65. The supervisory monitoring process itself retained host affinity; it did not perform the PPO computation.

## Verified outcome

All 65 final verification checks passed. A separate nine-check inspection of the emitted USD file confirmed the static cube collider, 4000×4000×0.2 m scale, z translation −0.1 m, z-up convention, friction 1, restitution 0, multiply combine modes and absence of a rigid-body API. Its SHA256 is `10e9371a277cb98e9135135785e9853688873da7be1bed1c04e9d8c0bc6f5391`.

| Quantity | Independent observation |
|---|---:|
| Original reference sources | 8,278 |
| Original reference frames | 4,825,012 |
| Parallel environments | 8,192 |
| Completed collection/PPO iterations | 10 / 10 |
| Actual optimizer updates | 200 |
| Environment transitions | 1,966,080 |
| Trace rows | 1 initial + 960 physics + 240 reward = 1,201 |
| Finite loss-history rows | 10 |
| Telemetry nonfinite counters inspected | 12,489 |
| Sum of those nonfinite counters | 0 |
| State/reward diagnostic alerts | None / none |
| Parameter maximum absolute change | 0.03848659247159958 |
| New checkpoints | 0 |
| Automatic checkpoint calls suppressed | 2 |
| Raw completed attempts, excluding synthetic initial resets | 47,368 |
| Peak Torch allocated bytes | 17,285,030,400 |
| Peak Torch reserved bytes | 18,358,468,608 |

Loss indices were exactly 0–9, with update counters 20,40,…,200. The private log independently contained all ten iteration indices and the completion sentinel. Every trace numeric value was finite, physics/reward indices and 4:1 alignment matched, and the trace SHA/byte length matched actual server bytes. The final value-function loss was 0.17335084602236747; finite losses and short-run parameter movement do not establish convergence or policy quality.

Observed maxima across the trace: joint velocity 61.45398712158203, body linear velocity 17.05051040649414, body angular velocity 92.25115966796875, absolute reward 0.9100826382637024 and absolute contact-force component 24,765.326171875. These remain measured diagnostics, not acceptance thresholds or evidence of physically acceptable motions.

After actual exit I independently reran the read-only full-library input inspection, rehashing the prepared/reference files and checking original/mirror and split bindings. Its complete aggregate result exactly matched the run receipt: 8,278 original sources, 4,825,012 frames and reference-order fingerprint `0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7`. All 18 frozen local/server file byte hashes remained unchanged. No process for this run remained active and the old generator child remained stopped.

## Initial-state and action equality

A separate CPU-only reader, with CUDA devices hidden and affinity0–3, loaded the original and newly saved private initial-state tensors after actual exit. All five keys (`motion_idx`, `time_steps`, `root_state_w`, `joint_pos`, `joint_vel`) matched exactly in values, shapes and dtypes; maximum differences were zero. The entire saved snapshot was also byte-identical: SHA256 `e445f357cb80d90c7b4ea58ced981716120074529c07889b9dfa862c5b29685a`.

The independently inspected, SHA-bound action-audit hook recorded selected first-action maximum difference 0 against the original alert environments. This is equality for that selected subset, not an independently stored first-action array for all 8,192 environments. No all-environment first-action raw snapshot exists.

Native PhysX root state is not bit-identical to the cached state: the receipt records global maximum absolute difference `7.62939453125e-06`; an independent comparison of the saved selected native roots to original roots measured `5.960464477539063e-08`. Native joint positions and velocities matched exactly. The author receipt has the same global native-root difference. These nonzero measurements are retained explicitly; no new tolerance, numerical setting or acceptance threshold was introduced. Exact cached initial-state equality must not be misreported as exact native-root equality.

## Shutdown, collection and privacy

The actual run used the documented 20-second headless shutdown fallback: `native_close_returned=false`, `teardown_fallback_used=true`, `shutdown_completion_verified=true`, after the successful diagnostic and closed output writers. The private stack identifies the simulation-stop/render path. It does not establish that `app.close` was reached or that native teardown returned successfully. Child exit0 is verified separately from native shutdown success.

Only after actual exit, the documented collector was executed with the document run name:

```text
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_contact_diagnostics.py --run solid_floor_ppo_document_v2 --family full_contact
```

Its three private files total 2,504,307 bytes. I rechecked their local bytes against the collector's per-file SHA256 manifest. A supplementary read-only fetch preserved the 4,364-byte native teardown stack with an independent SHA check. Private arrays, source indices and logs remain under `D:/具身智能/TextOp复现_20261024/datasets/contact_diagnostics_private_20261026/solid_floor_ppo_document_v2`; none were copied into public results. The public result directory contains aggregate receipts, trace and witness evidence only.

## Document-versus-reality qualifications

1. The numerical execution expectations matched. Corrected v2 source hashes identify actual local/server bytes; the prior document_v1 preflight failure remains untouched. This run required no executable, specification, configuration or numeric changes, no existing-run retry and no shared dependency modifications.
2. GPU residency is not exclusive to GPU1. A read-only process sample attributed 456 MiB on GPU0 and 10,382 MiB on GPU1 to the same SDK worker. The configured physics/Torch target is GPU1; the purpose of the smaller GPU0 allocation was not proven. `gpu_process_observation.json` preserves the UUID/index mapping and observation. The command was not changed to hide this behavior.
3. The collector's Windows `write_text` translates public JSON/JSONL newlines to CRLF. Consequently all four local aggregate byte hashes differ from their server LF hashes. Replacing CRLF with LF yields exactly the server SHA for each file. This is a transport serialization difference, not different numerical evidence. Private base64-backed copies retain exact bytes. Both local and server digests are recorded in `verification.json`.
4. The generic inherited `paper_settings.own_choices_not_specified_by_paper.floor` string still describes the baseline plane. The candidate's explicit `diagnostic_floor`, intervention field, frozen source and generated USD all identify the executed solid box. Treat the generic inherited string as baseline metadata, not the executed collider description; no receipt was edited.
5. Native teardown fallback and nonzero native-versus-cached root differences are disclosed above. The documentation explicitly requires this distinction. No claim of native-close success, whole-scene long-run stability or motion acceptance is supported.
6. Author and document final losses/parameter changes are not byte-identical despite identical initial snapshot/selected first action. The document did not require bitwise PPO trajectory repeatability. The observed author final value loss was 0.1899787500500679 versus document 0.17335084602236747. Both runs are short bounded diagnostics; do not promote this to a reproducible-convergence claim.

## Binding digests and evidence

| Binding | Actual server/local SHA256 where applicable |
|---|---|
| solid-floor PPO specification | `533e625204e45f3395798b6fe274de99434d2786060a5584a741618ea6b2730f` |
| PPO probe | `3129f2f071f9e333d3548d61b2278f9c650425a21402054e070ab321e7d6f28b` |
| Supervisor | `b59bcab493a7207953dc4e9208d8217356c409e0f7aa6786139a4b2197d9d417` |
| Capacity specification | `0f9dc72c4c616e673d343d5e8a49a091f48e40dc2cd9156d32a6e83854b19a40` |
| Paper step2 configuration | `d668360a8cbc1d4ad48845467841f3d4d9804f291a86790201ac3de74fec42dd` |
| Actual server receipt bytes | `597ddb5508264769bb7953a62697ad82120b0a84ad74291112467dd8952acf7d` |
| Actual server numerical trace bytes | `8616e04e0c8a49e37a7f335e58676d286ac460ed9dd68b4d2df36242fbb29ab1` |
| Actual server numerical summary bytes | `3c65dbd4dc6bbb99d8e9fb30a3d23e77539d86287868e07be19ab06f957423b3` |
| Actual server exit proof bytes | `8af7c65c170641dcc55d907e75fe3bb302fb24cfbb206549fd15182fae747c1e` |

Full bindings and evidence: `preflight.json`, `execution.json`, `launch_verified.json`, `postflight.json`, `verification.json`, `private_state_equality_aggregate.json`, `floor_and_shutdown_aggregate.json` and `gpu_process_observation.json`. Reproducible local verification helpers are retained alongside this report. Public copies are in `baselines/textop/results/full_contact_20261026/solid_floor_ppo_document_v2`.

The witness certifies execution of this frozen ten-iteration candidate and the stated measured checks. The 40,000-iteration policy, real two-pass physical filtering, accepted dataset, generator and full closed loop remain unexecuted by this witness. No main report was edited, and no commit, push, deletion, resume or formal launch was performed.
