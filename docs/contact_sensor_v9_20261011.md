# V9: passive humanoid foot-sensor observation

Read the unchanged .aris/compute/closd-baseline-20261010.md runtime ledger.
This adds four passive solver-constraint sensors to the same released SMPL
asset at L_Ankle, L_Toe, R_Ankle, R_Toe. Gravity/forward-dynamics forces are
excluded; readings use world coordinates. Articulated constraint forces may
include joint reactions: these sensors are NOT declared pure contact ground
truth and do not replace existing metrics or thresholds. Policy observation
mode remains self_obs_v=1, so sensor readings do not enter its observations.

Fresh document witness executes exactly once; no edits, retries or --run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sensor_v9_20261011.py --check
```

Expected exit0, CONTACT_SAMPLING_BATCH_PASS repeat_a and
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS. Independently inspect
results/contact_sensor_v9_20261011/raw/repeat_a: original 12 attempts retained;
120 frames if completed; seven reset state and four cache differences zero;
same generator/tracker/asset hashes, target/text/noise, effective SimParams;
four sensor bodies, observation mode1, sensor_metrics_used_for_selection=false;
sensor_diagnostic.npz arrays sensor[120,12,4,6], root[120,12,13], dof_pos/vel,
action all finite. Verify all source/protocol/version hashes and NPZ hash.

Root subsequently performs independent repeat_b via --run, comparing sensor,
root, DOF and actions and the V8 kinematics. Initial witness certifies only
recording, not force validity, repeatability or innovation effectiveness.
Box/plane fixture previously measured a 5kg object's support close to49.05N;
it does not validate these articulated sensors. Complete measurement gates
before any contact-based efficacy claim. Failures retained, no overwrite.

No installs, paid API, generator/tracker changes, deletion or Overleaf result
edits. All attempts count toward72GPUh total/4GPUh measurement. Require20GiB
data free and15GiB maximum new outputs. Runtime sources stay upstream-clean.
