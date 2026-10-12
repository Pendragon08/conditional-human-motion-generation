# V50 independent document-following witness

Status: **FAIL_RETAINED**. Date: 2026-10-12. The documented invocation was executed exactly once, exit **1**. No retry or repair occurred.

## Exact documented command

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/calibrate_surface_proxy_v50_20261012.py
```

Expected sentinel: `SURFACE_PROXY_ANALYTICAL_CONTROLS PASS_ANALYTICAL_CONTROLS 24`. It was absent.

## Failure and document divergence

The invocation created the remote `first_attempt` directory, then failed reading:
`results/native_reference_velocity_v46_20261012/public/raw/val_native_replay_s11/metadata.json`.

Read-only inspection confirmed that this remote metadata is missing and the retained `first_attempt` directory is empty. Both `summary.json` and `analytical_controls.npz` are absent. No controls were executed. A matching local-path metadata file exists, SHA256 `6a7b69db8f65aa8c1e6c1655f4c3300d8b34fa61dbf561c870b9e33d73c8c533`; it was neither transferred nor substituted.

The documented overlay runtime NumPy is **1.23.5**, matching the document, at `/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages/numpy/__init__.py`. Python is 3.8.20. The missing input stopped this invocation before numeric evaluation.

Provenance correction: the first read-only hash inspection omitted the documented PYTHONPATH and observed NumPy **1.24.4** in the bare venv. The earlier claim that this was a document/runtime version divergence was incorrect. A separate read-only environment probe with the documented PYTHONPATH confirmed overlay NumPy 1.23.5; it did not rerun V50. Both probe results are retained in JSON with their commands.

Corrected read-only environment probe (exit 0):

```text
env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -
```

Its stdin prints NumPy version/path and Python version. JSON preserves the exact SSH command and stdin source.

## Independently observed hashes

| Artifact | Remote SHA256 |
|---|---|
| V50 source | `e31006c9ffdda4db16b1da2fadac47558c674a5f34f4c65687437f90c5a3e826` |
| V46 metadata | Missing; unavailable |
| V46 frozen thresholds | `451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b` |
| Frozen V21 measurement source | `923ab0f00c0d1cddf005bc4b615e87737bc474376439f5fb1fbe67ff0ff14068` |
| NPZ arrays / summary | Absent; unavailable |

Available source/input hashes match their local copies. The document gives no expected hash constants, so this establishes identity between available local and remote copies, not historical immutability.

## All 24 planned controls

Local metadata names four boxes, each with eight points and zero radii. The source plans six controls per box. The following expected checks were independently reviewed against the source and thresholds (sliding speed 0.05 m/s, support force >10 N, height <0.02 m). Every cell is **not executed**.

| Box | Static | Flat sliding | Positive rocking | Negative rocking | Rocking plus sliding | Airborne |
|---|---|---|---|---|---|---|
| L_Ankle | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |
| L_Toe | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |
| R_Ankle | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |
| R_Toe | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |

For every supported control the planned predicate requires 60 support observations. Static and both rocking controls expect slip fraction 0; 0.2 m/s flat sliding and rocking plus translation expect fraction 1. Airborne expects support count 0 and slip fraction null. These are planned predicates, not observed results.

## Scope and provisional independence

Source inspection supports the intended scope: prescribed instantaneous rigid-body states, no generator/simulator/GPU entrypoints, prescribed support force, and parameter-sampled angles rather than integrated trajectories. The analytic pivot is selected from box bounds independently of the metric's argmin. However, construction reuses V21's `rotate_xyzw` and evaluates V21's `surface_metrics`. It remains **same-family provisional**; no independent contact-ground-truth evaluator validates the output.

This failed witness establishes neither the metric's 24 numerical responses nor native contact locations, measured force distribution, simulator rolling accuracy, human sliding, task semantics, downstream utility, or innovation confirmation. Angular speed alone cannot invalidate the proxy. No threshold or historical score was changed. Innovation counts remain three candidates, one first-round efficacy comparison, two incomplete, zero confirmed.

## Preserved execution output

```text
Traceback (most recent call last):
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/calibrate_surface_proxy_v50_20261012.py", line 86, in <module>
    main()
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/calibrate_surface_proxy_v50_20261012.py", line 27, in main
    meta = json.loads(meta_path.read_text())
  File "/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/lib/python3.8/pathlib.py", line 1236, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
  File "/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/lib/python3.8/pathlib.py", line 1222, in open
    return io.open(self, mode, buffering, encoding, errors, newline,
  File "/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/lib/python3.8/pathlib.py", line 1078, in _opener
    return self._accessor.open(self, flags, mode)
FileNotFoundError: [Errno 2] No such file or directory: '/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/native_reference_velocity_v46_20261012/public/raw/val_native_replay_s11/metadata.json'
```

No human trajectories were retrieved. No retry, package install, experiment fix, GPU execution, git change, or Overleaf write was performed. JSON contains the read-only inspection command and full per-control planned predicates.

