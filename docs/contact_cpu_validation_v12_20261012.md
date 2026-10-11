# Full validation after stable export-route change

Warm runtime .aris/compute/closd-baseline-20261010.md, canonical environment
81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780.
Use unchanged V11 collector and CPU-export/CUDA-mirror bridge, frozen models,
GPU PhysX, same120step reference clock and original targets. No installs.
V11 within-route two repeats are identical; V11 versus GPU export had small
first-step numerical differences amplified by feedback. Therefore this is a
new fixed experimental setting, not behavior-identical instrumentation. Do
not pool with older results or claim efficacy. All methods must be rerun.

Preselected full validation has24sources×3offsets=72 inputs. No sample selection
by quality. All failures and early terminations remain. Root continuation uses
planned/native each seeds11,12,13:432 attempts. Acceptance thresholds identical
to original frozen thresholds; no relaxing foot-slip or continuity gates.
If physical quality remains0 across all runs, stop the condition-compensation
route pending quality diagnosis; do not proceed to G1/downstream or call it
an innovation. Full known-good humanoid quality calibration is still required.

Fresh agent executes only this exact documented command once; no retry or --run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_cpu_validation_v12_20261012.py --check
```

Expected exit0 and CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS. Inspect actual
72×120 recordings, original-source manifest count24, three offsets, reset
consistency and same-time reference clock0..119. Every array must be finite,
terminal cases retained, frozen code/model/input hashes match. No predetermined
number of completed or successful episodes. Report actual rates without calling
them effectiveness. Record local results/contact_cpu_validation_v12_20261012/
DOCUMENT_WITNESS.md; licensed human arrays stay server-local. Root runs the
remaining five batches after witness recording. Free/data20GiB and output15GiB,
72GPUh global /4GPUh measurement including all prior attempts. Preserve existing
GPU jobs. No paid APIs, real robot or Overleaf experimental writes.
