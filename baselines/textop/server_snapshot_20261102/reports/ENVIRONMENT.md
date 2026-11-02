# TextOp training environment

Provider: nlp03, user ouyanghao, project `/data/code/ouyanghao/oyh/textop-reproduction-20261024`.
Spec: `configs/env-spec.json`; canonical spec hash is recorded in SERVER_NOTES.md.
The isolated Python 3.8 venv inherits the existing PBHC CUDA 11.7 / PyTorch 1.13.1 installation read only. It installs official pinned Python dependencies in an overlay. TensorRT is deferred to a separate deployment environment because the training path does not import it.

From Windows PowerShell, run this exact environment witness invocation:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/textop-reproduction-20261024 && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 env/bin/python scripts/kernel_witness.py"
```

Expected: exit 0, sentinel TEXTOP_KERNEL_WITNESS, NVIDIA RTX A6000, output [2,8,57], Python3.8.20, finite loss and finite_gradients=true. The configured_cuda_fraction field records the allocation limit requested from PyTorch; it does not measure total device utilization. This is a synthetic-input environment check, not a reproduction result. The witness writes its JSON receipt but does not modify datasets or launch training. Shared-base non-mutation is an installation procedure constraint, not a claim certified by this numerical witness.

Official training uses `scripts/train_official.py` from the `upstream/TextOpRobotMDAR` working directory. Official defaults are retained with documented USAGE.md overrides: stages [100000,50000,50000], batch512, four primitives, history2/future8, weighted sampling and rollout enabled, seed0. Correct the action_statistics path to the actual public dataset directory. The wrapper counts effective optimizer updates separately from official loop steps and preserves the official behavior of skipping nonfinite gradients. It logs each skip, saves and stops only after 20 consecutive skips, adds an extra step1000 checkpoint and a 1GiB disk guard; it does not alter the model, losses or update schedule. Attempt1 stopped at loop644/effective643 because the initial wrapper incorrectly treated a single official skip as fatal; its records are preserved separately.

Only launch after every required public dataset and robot description passes full SHA256/size validation. The official released pretrained model uses PRIVATE-DATA normalization and must not be mixed with the public BABEL statistics used for this training.

PhyGile/ReactiveBFM remain paused. Resources: GPU1; 65% CUDA allocator limit; four CPU threads. No physical robot or GUI required. Raw data/checkpoints stay outside the public Git tree.
