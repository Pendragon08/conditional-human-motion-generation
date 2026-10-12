# V45 original-reference calibration after the reference velocity repair

Replay the already bound V33 validation dataset_reference.npz (72 cases, 24
sources and three offsets) once through the fixed V43 interface. No generator
sampling, new target, threshold, tracker, or source selection. No test inputs.
All failures remain in the denominator. Zero-offset cases are additionally
reported, without discarding the other offsets. This diagnostic is not candidate
efficacy and does not certify MoCap-derived references as executable ground truth.

Prerequisite: V43 independent witness PASS, exact original reference hash and
all frozen sources. Strict execution-time lineage checks and the shared current
resource ledger apply: measurement <=4 GPUh, total <=72 GPUh, data free >=20 GiB,
new output <=15 GiB. No automatic retry; existing incomplete output is rejected.
The V43 collector supplies exactly successive applied reference velocities.

Execute exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_original_reference_velocity_v45_20261012.py --run
```

Expected summary/state COMPLETE_ORIGINAL_REFERENCE_VELOCITY_DIAGNOSTIC. An
independent document-following witness must check retained cases, dataset input
hash, unchanged source goals, generation_batch_calls=0, canonical position
identity, recorded velocity rule, native float32 time alignment, and all bound
receipt/output hashes. Report divergences rather than rerunning. A nonzero
quality count does not validate C2/C3; a zero count does not prove that all
generated motion is infeasible. Do not change old scores, Overleaf, thresholds,
or use this result to bypass efficacy gates.
