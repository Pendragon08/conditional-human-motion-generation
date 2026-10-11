# Independent V24 document-following witness

Verdict: **FAIL_RETAINED — collection completed; metric certificate failed.** This is a measurement-validation witness, not innovation efficacy.

The exact command in `docs/contact_com_validation_v24_20261012.md` was executed once, without edits, retries, added simulation, package changes or Overleaf writes:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_com_validation_v24_20261012.py --check
```

Live handle: `36694`, subsequently terminal with exit **1**. Native collection printed `PHYSICS_GUIDED_COLLECTION_PASS 72 0`. The documented `CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS` sentinel was absent. State is `FAILED_RETAINED`. Failure occurred after collection in `certify_cached` / `compare_metrics`: `Metric differs beyond float64 roundoff: metrics[0].surface_slip_proxy_mean_m_s`.

## Read-only checks

- All 23 declared source/config/document hashes match; V19 COM calibration and V22 articulated calibration hashes match. Generator and tracker hashes match. Manifest binding and unchanged frozen thresholds verified.
- All saved V24 trajectory arrays (actual, reference, quaternion, linear/angular velocity, contact force, valid, terminated, reference frame) are finite and **exactly equal** to V12 planned seed 11. All sensor/root/DOF/action arrays are likewise exactly equal, maximum difference **0**. This passive readout caused no observed trajectory/action change for the 72-input batch.
- Original goal and generation-noise hashes equal V12. Seven reset-state maximum differences are 0, reset gate passes, and reference clock is exactly 0–119.
- 284,941 native contact records; no parsing errors. Native fields include camel-case `localPos0` / `localPos1`, as handled by the normalized field adapter. Native records cover environments 0–71 and steps 0–119. Body COM array is finite with shape 72 x 24 x 3. Native JSON hash matches metadata.
- Independent metric import resolves to the intended server `scripts/physics_guided_metric_v23_20261012.py`, bound SHA256 `b05b6ed8053ac037432471dabdf8608da383c76d4d2375ce10823e003f232c69`.
- Independently recomputed classifications equal saved classifications: 70 complete, 23 condition pass, 0 physical pass, 53 tracking pass, 0 joint pass. Two early terminations retained. Negative classifications do not explain invocation failure: numeric certification does.

## Numeric discrepancy

For case `008349:-0.15`:

| Measurement | Saved V24 | Independently recomputed V23 | Historical V12 |
|---|---:|---:|---:|
| Mean slip proxy (m/s) | 0.07545065144151883 | 0.07545065141555918 | 0.0880626921869827 |
| Slip fraction | 0.33557046979865773 | 0.33557046979865773 | 0.42953020134228187 |

Mean-slip disagreement is 2.5959651095419645e-11 m/s for this case. All 72 saved/recomputed mean-slip values differ; maximum discrepancy is 3.7682390541249333e-10 m/s. Some heading scalars also differ by 4.44e-16 rad. Array dtypes are recorded in the independent JSON. This witness has not proven the reason for the small discrepancy; no tolerance, score, threshold or source was changed. The verifier failure remains authoritative and requires diagnosis before continuation.

`document_witness_check.json` contains compact independent bindings, field/dtype/invariance checks, case 0 comparisons and model hashes. No raw human trajectories or contact manifolds were copied locally. Original files and failure state remain untouched on the server.