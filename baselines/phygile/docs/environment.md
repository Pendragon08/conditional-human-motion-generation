# PhyGile environment ledger

Spec hash: `fa5883c8`.

Provider: existing authorized SSH server, NVIDIA RTX A6000 GPU 0. Existing idle graphics allocations are shared; do not stop existing jobs.

Reuse Isaac Sim 4.5.0 and Isaac Lab 2.1.0 without modifying their installation. Override `ISAAC_SIM_ROOT` and `ISAAC_LAB_ROOT` to relocate. The tracking package is resolved from this directory's `upstream/source/whole_body_tracking`, never from the old experiment checkout.

No pip installation is needed on the current host. Re-run version and GPU witnesses after dependency changes. Spec `fa5883c8` passed a seeded GPU forward/backward witness and an independent document-following 32-environment, one-update simulation on 2026-10-17; see `results/document_witness/report.md`. The independent repeat after this ledger edit is recorded in `results/document_witness_recheck/report.md`. Isaac Lab 2.1.0 was additionally verified from the installation's VERSION file.

Local server work directory: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017`.

Documented kernel invocation from the work directory:
```sh
bash scripts/run_isaac.sh scripts/kernel_witness.py
```

Documented simulation smoke invocation (32 environments, one PPO update, GPU 0):
```sh
bash scripts/smoke.sh
```
