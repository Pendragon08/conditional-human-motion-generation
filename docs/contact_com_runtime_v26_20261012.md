# V26: numerical-runtime-only certification repair

## Purpose and preserved failure

V24 collected 72 planned-prefix, seed-11 validation inputs successfully, but its wrapper imported NumPy 1.24.4 while the collector used the existing closd-r3 overlay (NumPy 1.23.5). V25 read-only recomputation found maximum mean-slip difference 3.7682390541249333e-10 m/s in the wrapper environment and exactly zero in the collector environment. Classifications did not change. This is a verifier runtime defect, not candidate efficacy.

No package install, simulation change, policy/generator change, tolerance widening or resimulation of the first completed 72 is permitted. Original V24 versions, bound requests, lineage, metadata, metrics, contacts and trajectories remain unchanged. The original failed state and budget/versions are copied into the V26 directory before V24's resumable state advances. V24 independent failure report remains negative and unchanged.

## Exact fresh-document check invocation

Run this command once from local PowerShell; do not retry on failure:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_com_runtime_v26_20261012.py --check
```

Expected exit 0, existing V24 `CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS`, followed by `CONTACT_COM_RUNTIME_DOCUMENT_CHECK_PASS`. V26 `runtime_check.json` must show NumPy 1.23.5 from closd-r3, `preserved_outputs_unchanged=true` and identical before/after budget batch counts. The cached batch must be certified using unchanged V23 metric and unchanged V24 numeric comparison; all original requests, metadata, goals, thresholds and trajectory hashes must match. Independently inspect the preserved failure snapshot and numerical certificate; do not interpret a passing wrapper check as physical quality or innovation success.

Save independent report to `results/contact_com_runtime_v26_20261012/DOCUMENT_WITNESS.md`. Record exact command, terminal exit, no new simulation, source/document/output SHA checks and any divergence. Do not repair or rerun in witness role.

## Authorized continuation after witness

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_com_runtime_v26_20261012.py --run
```

It resumes V24 through unchanged versions and collector, only adds missing five batches, retains all source denominators/failures, charges their time to the existing measurement budget, and keeps data/output guards. Expected final `CONTACT_COM_RUNTIME_FULL_VALIDATION_COMPLETE`, V24 validation summary 432 attempted inputs, six batches. A physical-quality floor stops the condition-feedback route; never relax thresholds to continue G1/downstream. This scope does not certify actual rolling-contact ground truth or semantic compliance. No Overleaf experiment writes.
