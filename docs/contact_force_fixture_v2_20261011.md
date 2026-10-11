# PhysX contact calibration with known box/plane fixtures

Use the unchanged warm Python/IsaacGym runtime documented in
.aris/compute/closd-baseline-20261010.md. This test has no motion generator,
tracker, humanoid articulation, randomized dynamics or real robot. Four 5kg
boxes on a frictionless plane test resting support, known horizontal sliding,
free fall and upward takeoff. Net contact forces and constraint-only force
sensors are recorded together. Sensor gravity/forward dynamics excluded.

The analytical checks are fixed before execution: resting surface height within
1cm, resting speed below2cm/s; settled net and sensor normal forces each within
10% of measured actor mass times9.81; sliding velocity within2cm/s of0.2m/s;
initial20frames of high drop have height above0.5m and force below1N. These are
fixture calibration tolerances, not new thresholds for humanoid acceptance.

Independent witness executes the exact command once; no edits, retries or --run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_force_fixture_v2_20261011.py --check
```

Expected exit0 and FIXTURE_DOCUMENT_EXECUTION_COMPLETE repeat_a. Read
results/contact_force_fixture_v2_20261011/repeat_a/summary.json; record every
check, PASS or FAIL_RETAINED. The wrapper's completion sentinel only means that
measurements were saved, not that the physics calibration passed. Check recorded
120x4 states, net-contact arrays and constraint-sensor readings, finiteness,
mass, source/output hashes and all five bound source/config/doc files.
Write local results/contact_force_fixture_v2_20261011/DOCUMENT_WITNESS.md.
After witness, root executes --run for repeat_b if instrumented recording works.

All failed attempts retained, budgeted within original72GPUh/4GPUh measurement,
20GiB free-space minimum and15GiB total new output cap. No installs/deletions,
paid API, hardware tests or Overleaf result edits. A box result cannot establish
human contact-point truth, articulated sensor validity or motion-quality gates.

## V2 hypothesis

Clear only the derived net-contact export tensor before native simulate, then
run native simulate/fetch/refresh unchanged. No actor force setter, simulator
parameter, asset or sensor change. Add the physically required frictionless
horizontal force below1N check. Failure is retained; never treat erased contact
as correction: known49.05N static support and airborne checks must still pass.
After two runs, compare state and constraint sensors to the original fixture
arrays before accepting any instrumentation effect. Not an innovation test.
