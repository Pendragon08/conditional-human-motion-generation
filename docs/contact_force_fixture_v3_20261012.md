# CPU tensor export / unchanged GPU PhysX box calibration

Use the unchanged warm runtime and canonical environment SHA in
.aris/compute/closd-baseline-20261010.md. No installs, generator, tracker,
controller or humanoid. GPU0 must be idle with existing viewer contexts
preserved; free /data >=20GiB. Original budgets72GPUh/measurement4GPUh,
new outputs15GiB apply, including failures.

V2 export clearing did not resolve repeatability; its failures remain retained.
V3 changes only use_gpu_pipeline=False. PhysX still runs on GPU with the
same gravity, friction, mass, timestep, substeps and solver settings. No
tensor clearing or actor force setter. CPU tensor export is a diagnostic,
not an accepted replacement for the humanoid experiment yet.

Fresh witness executes exactly once, without edits, retries or --run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_force_fixture_v3_20261012.py --check
```

Expected exit0, FIXTURE_DOCUMENT_EXECUTION_COMPLETE repeat_a. Completion is
not calibration PASS. Recompute all nine checks from actual arrays: settled
height within1cm, speed below2cm/s, net/sensor support within10% of49.05N,
sliding speed within2cm/s of0.2m/s, first20 airborne frames height above0.5m
and net/sensor below1N, all frictionless horizontal net forces below1N.
Record source/config/doc hashes, raw arrays hash and every failed check in
results/contact_force_fixture_v3_20261012/DOCUMENT_WITNESS.md locally.
Do not modify scripts or call --run. Root performs repeat_b and compares
state and sensors against V1/V2 afterward. No human/innovation efficacy claim.
