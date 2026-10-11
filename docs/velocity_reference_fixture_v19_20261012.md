# V19 independent rigid-body velocity-reference calibration

An isolated cube with zero COM offset and another with 6cm COM offset rotate at 3rad/s in zero gravity. No controller, human data, contact or generator. CPU rigid-body tensors, GPU PhysX,240steps at240Hz. Central finite differences of body origin and transformed COM are compared with exported linear velocity. Preregistered: centered residual below3mm/s for the supported hypothesis, above5cm/s for the other, zero-offset control below3mm/s and angular rate constant within.001rad/s. Do not apply either interpretation to humanoid metrics until this separate fixture passes. Negative result is preserved.

Fresh witness executes exactly once, only when no V17 process is live:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_velocity_reference_fixture_v19_20261012.py
```

Wrapper checks spare GPU, >=20GiB data free,15GiB outputs and remaining72GPUh/4measurementGPUh. Source/spec hashes and every attempt are logged; no retry or overwrite. Expected exit0 and `VELOCITY_REFERENCE_DOCUMENT_EXECUTION_COMPLETE`, followed by PASS or retained FAIL and velocity reference interpretation. Physics quality of a humanoid, real forces and innovation efficacy are outside scope. No Overleaf writes or installations.
