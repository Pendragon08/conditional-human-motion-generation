# Human CPU-export / CUDA-mirror measurement diagnostic

Warm environment: .aris/compute/closd-baseline-20261010.md and
configs/env-closd-baseline-20261010.json, canonical SHA256
81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780.
No install or package/spec change. Models and policy observations remain CUDA,
frozen weights,574 observations/69 actions/self_obs_v1. Only native simulation
exports move to CPU. A bridge mirrors exported tensors to CUDA at the same
existing acquire/refresh points and converts setter inputs to CPU unchanged.
The GPU PhysX solver, dt, substeps, body geometry and existing completion fence
remain unchanged. Passive foot sensors are retained as secondary evidence,
never substituted for ground contact. No metric or threshold change.

Precondition: box V3 two-run comparison supports CPU export with identical
state/sensor to V1 and zero repeat net-force differences. This does not imply
human contact correctness. V11 uses4 validation sources×3offsets×120frames,
seed11, first attempts only. Failing attempts remain; no retry or overwrite.
Resources: idle GPU0 with prior viewer contexts preserved, /data free20GiB,
outputs15GiB,72GPUh total and4GPUh measurement including all prior attempts.

Fresh agent executes exactly once, no edits/retries/--run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_cpu_export_v11_20261012.py --check
```

Expected exit0 and CONTACT_SAMPLING_DOCUMENT_CHECK_PASS. This certifies recording
only, not measurement/efficacy. Check raw12×120 arrays, all finite, correct
reference time0..119; seven exported bootstrap vs first-action states identical,
four cache comparisons remain metadata-only. CPU native/GPU mirror descriptors
and sim metadata must say CPU pipeline/GPU PhysX. Verify frozen files/models and
output hashes, no policy observation or sensor-metric changes. Record all failures
in local results/contact_cpu_export_v11_20261012/DOCUMENT_WITNESS.md. Licensed
human NPZs remain server-local. Root separately runs repeat_b and compares
state, action, sensor and generation noise against V10 and between repeats.
If those differ, do not call it measurement-only repair. Full physical quality
and innovation gates are still unproven. No Overleaf experimental-result writes.
