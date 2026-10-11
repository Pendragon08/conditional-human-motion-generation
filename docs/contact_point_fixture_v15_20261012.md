# Native contact API availability in known box physics

Warm runtime .aris/compute/closd-baseline-20261010.md; no installs. Retain the V3
box fixture physics, solver, CPU export and GPU PhysX. Only add a read-only
get_env_rigid_contacts call after the same completed step. Four fixtures:
stationary box, frictionless0.2m/s slide, high airborne drop, upward takeoff.
This is availability/known-physics calibration, not innovation efficacy.

Fresh agent runs this command exactly once; no retries or extra simulations:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_fixture_v15_20261012.py
```

Expect exit0 and NATIVE_CONTACT_FIXTURE_DOCUMENT_EXECUTION_COMPLETE; the native
API may be unavailable, and this must be reported as a retained negative result,
not PASS. Inspect native_contacts.json, underlying log and saved state arrays.
If available, verify nonzero stationary plane contacts, zero initial airborne
ground force, field finiteness/body IDs, and read-only state equivalence versus
V3. BundledSDK says body-1 denotes ground; local_pos excludes thickness, offset
locates friction surface. Do not equate feature positions with friction points.
Do not assume last-substep manifold forces equal all-substep export forces.

4GPUh measurement and72GPUh total limits still apply, /data≥20GiB and new output
≤15GiB. Preserve existing workloads and old files. No APIs, model training,
human data publishing, score changes or Overleaf writes. Record independent
DOCUMENT_WITNESS.md under results/contact_point_fixture_v15_20261012 locally.

V14 failed on its second step because a contact-record loop variable shadowed
the exported contact tensor. Preserve V14 source/logs/ledger. V15 renames only
that local record variable. No simulator, state or thresholds change.
The current budget snapshot includes V14 failed wall time.
