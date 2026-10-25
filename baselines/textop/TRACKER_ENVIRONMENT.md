# TextOp tracker backend validation

Scope: reuse existing Isaac Sim4.5.0 / Isaac Lab project2.1.0 read only. The
IsaacLab package distribution identifies itself as0.36.21. Python3.10.15,
Torch2.5.1+cu118, NumPy1.26.0, Gymnasium0.29.1, RSL-RL2.3.1 are pinned by
configs/tracker-env-spec.json, canonical hash893388d9. No packages were installed.
The shared BeyondMimic G1 asset SHA256 is
dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b.
The exact TextOp author asset hash is not published; do not claim equivalence.

Resources: headless GPU1, CPU4, Torch allocator fraction0.65; preserve other
processes. The user explicitly authorized headless/shared-GPU reproduction.

The author execution produced TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED, real32
physics steps in8 G1 instances, exact29-joint teleport mapping, and seeded
557→2048→1024→512→29 MLP finite gradients. It constructed no optimizer.
The author's initial detached launcher did not retain a durable child exit code;
that exit status remains unknown. The independent direct invocation below must
also return exit0 before recording this backend as document-validated.

Run this exact PowerShell command once. The output directory must not exist.
Do not repair, remove files, retry with different flags, or launch full training.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/run_tracker.sh /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/tracker_kernel_witness.py --spec /data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/tracker-env-spec.json --output /data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/document_v1 --device cuda:1 --headless'
```

Expected: exit0 and TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED. The output receipt must
have status TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED, dof29, num_envs8,
physics_steps32, optimizer_updates0, physical_filter_executedfalse,
full_filter_training_readyfalse, ground_presentfalse, and
tracking_success_evaluatedfalse. Body count30 is expected after the URDF importer
merges fixed bodies. The receipt source SHA must match the author source
b1bcaee5476f4a8d7c660c411f4f2765c13a08889d6755cc29bf71e7fe3b7593.

This is a free-fall backend/kinematic diagnostic with finite motor responses and
no ground. It is not a contact test, motion success rate, physical filter,
trained tracker or full TextOp result. Keep reference_fk.private.npz and generated
USDs on the server; only aggregate receipts/logs are eligible for GitHub.
