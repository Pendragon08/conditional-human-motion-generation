# V24 full validation after COM metric amendment

V19 proves rigid-body exported linear velocity refers to COM; V21 corrects material-point velocity, verified against independent240Hz finite differences and preserved static/slide/airborne box controls. V22 must pass known articulated support/slide/air controls before launching. V23 applies the same corrected formula using native per-environment, per-body COMs. No tolerance, threshold, goal, policy or physics changes. Historical V12 scores remain intact and are not pooled. This is measurement calibration/quality diagnosis, not candidate innovation efficacy.

Fresh document-following witness executes exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_com_validation_v24_20261012.py --check
```

Expected exit0 and `CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS`. One planned-prefix batch,24 validation sources x3offsets=72 inputs, seed11. Verify exact source/calibration/model/config hashes, unchanged original goals and noise,120frame clocks, reset gate, finite saved arrays/native body COM properties, no native parsing errors, and recomputation against V23 corrected metric. Read-only compare arrays with V12 planned seed11 to ensure measurement readout changed no actions/states. No raw human motion/native manifolds published.

Owner continuation after fresh witness: replace `--check` with `--run`. Certify cached first batch, run remaining five planned/native batches at seeds11,12,13, preserving all432 attempts and early terminations. Guard72GPUh total/4measurementGPUh,15GiB round outputs,20GiB data free and spare GPU. Include all previous ledger entries. Original0.05m/s slip speed and0.10 slip-fraction thresholds stay fixed. No retries, installs, added IK, Overleaf or model training. If all joint/physical scores still zero, stop the condition-compensation efficacy route and retain the negative diagnosis; do not start G1/downstream or claim efficacy.
