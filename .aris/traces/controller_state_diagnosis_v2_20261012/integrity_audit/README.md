# Controller-state diagnosis V2 integrity-audit trace

This directory is the full read-only trace for the fresh same-family, provisional experiment audit dated 2026-10-12.

## Audit boundary

- Local project root: `D:/具身智能/conditional-human-motion-generation`
- Server project root: `/data/code/ouyanghao/oyh/conditional-human-motion-generation`
- Server raw arrays remained in place and were never copied locally.
- CPU Python: `/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python`
- `PYTHONPATH`: `/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages:/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts`
- `CUDA_VISIBLE_DEVICES` was empty for both successful audit commands.
- The audit did not invoke `--prepare`, `--check`, `--run`, a simulator, generation, or training.

## Files

- `independent_recompute.py`: independently authored NumPy recomputation. It does not import the experiment metric module.
- `independent_recompute.json`: successful structured result, status `PASS`.
- `independent_recompute_stdout.txt`: successful stdout; byte-identical to the structured JSON.
- `independent_recompute_exit.txt`: successful exit record (`0`).
- `independent_recompute_attempt1_stdout.txt`: preserved initial auditor failure caused by assuming the fixed reference was case-major rather than time-major. No experiment file was changed.
- `independent_recompute_attempt1_exit.txt`: malformed first exit capture (`\`) caused by a shell-quoting error; the traceback in the paired stdout is the authoritative failure evidence.
- `independent_recompute_attempt2_stdout.txt`: preserved second auditor failure caused by omitting `_r0` from the historical source directory name. No experiment file was changed.
- `independent_recompute_attempt2_exit.txt`: exit `1`.
- `publication_guard_recheck_stdout.json`: independent read-only server rerun of the supplemental publication binding guard.
- `publication_guard_recheck_stderr.txt`: empty on success.
- `publication_guard_recheck_exit.txt`: exit `0`.

## Successful recomputation command

```text
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages:/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python .aris/traces/controller_state_diagnosis_v2_20261012/integrity_audit/independent_recompute.py --output .aris/traces/controller_state_diagnosis_v2_20261012/integrity_audit/independent_recompute.json
```

The successful output reports zero metric differences, zero binding failures, zero frozen-criterion differences, and zero source-hash mismatches. It also reconstructs the fixed tracking reference exactly from the earlier simulation's saved `actual` trajectory plus terminal padding.

## Supplemental publication-binding recheck

```text
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages:/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_publication_guard_20261012.py --server
```

Exit was `0`; `raw_array_bytes_verified` was `true`; no server-only arrays were unavailable. The recheck is semantically identical to `results/controller_state_diagnosis_v2_20261012/PUBLICATION_BINDING_SERVER.json`.

## Verdict boundary

The deterministic artifact check is PASS. The experiment-integrity verdict is WARN because tracking is a controller-derived synthetic proxy, endpoint timing is a near-minimum-distance timing proxy, evaluation scope is one seed over an internal train/test mix for states of one algorithm, and the final parent completion lacks an automatic server-generated receipt. See `results/controller_state_diagnosis_v2_20261012/EXPERIMENT_AUDIT.md` and `.json`.
