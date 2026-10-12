# V42 validation-only reference and observation diagnosis

This is a passive instrumentation diagnostic, not C2/C3 efficacy. It retains all
24 validation sources and all three frozen offsets: 72 inputs, generation seed
11. Two paid measurement batches first generate planned-prefix references, then
replay those exact references. No test source, new controller, model tuning,
threshold change, new task or Overleaf result write is involved.

Record the returned reference velocity and previous pose from the upstream
generation cache, plus the raw task observation immediately before each action.
Do not call it the normalized policy observation. Recordings and licensed arrays
stay on the server; publish only summary, configuration, hashes and logs.

Run the documented command exactly once. The launcher binds source versions
before execution, accounts both batches against measurement/total ceilings and
enforces free disk/output limits. Existing viewer contexts can remain on idle
GPU0; active workloads are neither killed nor displaced. Missing or retrospective
lineage, changed hashes and incomplete attempts are rejected, never repaired or
silently restarted. Functional logs and every failed attempt remain intact.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reference_observation_v42_20261012.py --run
```

Expected `COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC` summary and terminal
`COMPLETE_INTERFACE_DIAGNOSTIC`. A fresh document-following witness must report
any divergence, including failures. Prior C1 remains negative; this diagnostic
must not count as another validated innovation. At the segment boundary,
generation uses its reconstructed previous frame whereas fixed replay uses the
preceding saved reference; compare the actual velocities before attributing any
behavior difference to this interface.
