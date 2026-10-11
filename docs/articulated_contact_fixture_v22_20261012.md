# V22 articulated support/slip controls

Known three-link fixed-base robot: foot box20x20x10cm with3cm COM offset on a driven horizontal prismatic joint and a free vertical prismatic joint. Three environments: stationary ground support, frictionless commanded20cm/s ground slide, and foot1m above ground. Existing V21 COM-corrected material-point proxy uses the original frozen thresholds.120steps60Hz, first60steps settle. No generator or learned controller. Native ground-manifold availability, no airborne contact, positive support, static slip fraction0 and slide fraction1 are preregistered; measured slider velocity must stay within2cm/s of command. The vertical DOF is unactuated and allows gravitational settlement onto the plane. Preserve any failed calibration.

Fresh witness executes the exact invocation once after V20 is terminal:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_articulated_contact_fixture_v22_20261012.py
```

Expected exit0 and `ARTICULATED_CONTACT_DOCUMENT_EXECUTION_COMPLETE` plus PASS or retained FAIL. Read-only independently recompute fixture fields, compare thresholds and source hashes. This does not validate all rolling contacts, real forces or any innovation. Budget/free-space guard:20GiB data free,15GiB study outputs,72GPUh total/4measurementGPUh; old jobs untouched. No retries, installation, Overleaf or human raw-data publishing.
