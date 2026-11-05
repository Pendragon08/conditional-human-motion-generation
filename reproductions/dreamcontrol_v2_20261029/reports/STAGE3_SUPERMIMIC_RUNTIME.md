# SuperMimic training-logic / existing PhysX runtime witness

Runtime ledger: server `/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103/COMPUTE_LEDGER.md`. Read it and `runtime_bridge_spec.json` first. Canonical spec hash is `c7724ac5`.

The already configured IsaacSim4.5/Python3.10/Torch2.5.1 runtime runs the calibrated PhysX Pick task. TextOp's pinned SuperMimic source provides unchanged MLP and running-normalization definitions; new112-state dynamics,31-action live policy, learned reward/termination heads and BPTT connect that algorithm to the existing robot, binary hands and object. This is an explicit project adaptation. TextOp's separate Python3.12/Newton/MuJoCo-Warp scene and its assets are retained. No dependency installation or shared source mutation occurs.

Scope: real4-environment64-control-step data collection,16 autoregressive world-model updates,1 imagined BPTT policy update, and full model/optimizer checkpoint round trip. Sole Actor input is the same102 live observation. This is a bounded implementation witness; it does not complete the training budget or establish task success.

Execute this exact command ONCE from the Windows client. Do not alter, repair, or repeat a failed/unknown run. Report original exit status, sentinel, output hash and any divergence. Never signal other tasks.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 10.147.17.32 python3 -B /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103/src/stage3_supermimic_doc_witness.py
```

Expected: exit0 and `SUPERMIMIC_PHYSX_DOC_WITNESS_PASSED`; the returned status JSON must say integration_smoke_completed,64 real vector steps,16 world-model updates,1 policy update,102 observations,31 actions, live-observation error below1e-4 and all six checkpoint state round-trip comparisons true. Original launch/child/exit evidence is under `doc_witness_launch_v1`; output is under `reports/physx_model_doc_witness_v1`. Wrapper enforces a180-second child timeout and refuses existing output paths. The helper restricts PyTorch allocation to8% of GPU1 and preserves the existing disk floor; CPU thread limits are two. Read failure evidence if the command refuses or times out; never assume success from an existing environment or import.
