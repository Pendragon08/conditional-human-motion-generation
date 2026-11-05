# Native universal tracker environment witness

Provider: existing SSH alias10.147.17.32. Warm-reuse TextOp's Python3.12/Torch2.11.0+cu126/IsaacLab3/Newton/MuJoCo-Warp environment. No installs or TextOp writes. Canonical server spec at R/stage3_supermimic29_20261103/native_universal_20261104/environment_spec.json has hash32a40546 (R=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029).

This fresh-agent check runs one seeded8×8 real CUDA multiplication on physical GPU0. It starts no simulator, policy optimizer, training, downloads, or foreign signals. New outputkernel_freshdoc_v1 must not already exist; refusal is reported rather than bypassed. Per-job12GiB GPU/32GiB RSS and project24/64GiB aggregate, runtime data floor8GiB+356MiB apply. Cache is a new private named directory under/dev/shm. No old task may be interrupted or changed.

Execute exactly once:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 10.147.17.32 python3 -u /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103/src/native_tracker500_guard.py --mode kernel --label kernel_freshdoc_v1
```

Require original SSH exit0, guardstatus passed and workerstatus kernel_passed, seed20261104, shape[8,8], RTX A6000, Torch2.11.0+cu126, spec_hash32a40546, finite seeded_kernel_error<1e-4. Report actual outputs and any mismatch. Do not retry, edit, install, run extra benchmarks or declare a task ready. This is environment validation only; the original fulltrain.py compiled/batched task test is separate.
