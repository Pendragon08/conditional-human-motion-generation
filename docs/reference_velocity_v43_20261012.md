# V43 applied-reference velocity consistency amendment

V42 measured a generation/replay interface discrepancy. V43 is a separate
measurement amendment, not an innovation, efficacy experiment or replacement of
historical results. Same 24 validation sources, three offsets, seed11, two paid
72-input diagnostic batches; retain all failures. Generator and controller
weights, goals, text, duration, position stream and acceptance thresholds stay
frozen. No test data, API, training, hardware or Overleaf result writes.

Reference velocity is the finite difference of successive positions actually
supplied to the tracker, divided by the native interval. Frame0 has no earlier
applied reference and uses zero velocity. Repeated queries of a frame reuse its
previous position; skipped frames or changed repeated positions fail. This
removes a reconstructed generator-context predecessor from the boundary velocity
calculation. Log exact velocity, previous position and raw task observation.

Execute this exact invocation once; no retries or source changes after execution:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reference_velocity_v43_20261012.py --run
```

Expected terminal `COMPLETE_INTERFACE_DIAGNOSTIC` and summary
`COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC`. The strict V42 launcher accounts
all current ledgers, guards disk/outputs and execution-time cache lineage.
Historical V42/V28 remain intact. An independent document-following witness must
report failures, check the new velocity against the applied position stream and
compare generation versus replay at all jointly valid frames. If states still
diverge, report it rather than claiming consistency was solved. No efficacy gate
is bypassed and no improvement is counted as a scientific innovation.
