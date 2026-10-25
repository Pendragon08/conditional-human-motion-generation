# 69D 数值验证环境

Spec: configs/paper69-env-spec.json, canonical hash934d8221, file-byte SHA25630bcaa928d4e7ed4cf7359d88dfe909334718957fffbd07bc98510467612d421.

Warm reuse of validated textop9d7f370d. No package installation, shared-environment modification, dataset replacement or training occurs. Python3.8.20, Torch1.13.1+cu117, MuJoCo3.2.3, NumPy1.24.4, Hydra1.3.2, OmegaConf2.3.0, OpenAI CLIP1.0. Uses existing authorized G1 XML and an existing official ViT-B/32 weight whose SHA256 is checked against the installed CLIP catalog; nothing is downloaded by the witness.

Execute this exact command from Windows PowerShell for the independent document-following validation:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/textop-reproduction-20261024/env/bin/python /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/paper69_witness.py --spec /data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/paper69-env-spec.json --output /data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/paper69_doc_v1.json --device cuda"
```

Expected exit0 and TEXTOP_PAPER69_NUMERICAL_WITNESS_PASSED. It reconstructs two already-prepared candidate fixtures, compares differentiable38-body FK with MuJoCo on composed root rotations and varying joints, runs unchanged official9-layer VAE and8-layer LDM forward/backward, verifies all six wrist dimensions carry geometric gradients, checks frozen VAE/CLIP during LDM, and generates two finite8-frame feature segments using exact last2 history reuse and five-step DDPM sampling. It checks the LDM rollout cap0.8 independently from VAE cap1.0.

The receipt includes source/asset hashes, codec and FK errors, gradient counts, output shapes and actual peak CUDA allocation. GPU1 uses the previously authorized65% allocator cap and four CPU threads. This small batch2 validation is not a training-memory estimate.

The implementation is process-local in scripts/paper69_runtime.py: a differentiable69D codec, a MuJoCo-topology quaternion FK with fixed-body mappings, correct qpos quaternion ordering,50Hz velocity differencing, LDM rollout cap, and int64 timestep indexing compatibility. Fixed upstream files are preserved. The geometric body set explicitly includes all38 non-world frames in the pinned GMR XML; the paper does not publish its final69D geometry-body selection. The final finite-difference velocity repeats the last difference at the sequence boundary. These choices are documented self-implementation details, not newly discovered author configuration.

Sampling uses cosine5-step DDPM, clean-latent prediction and CFG5. FIXED_LARGE matches beta_t variance at the noise-bearing reverse steps described by the paper; the terminal step adds no noise. The published FIXED_SMALL configuration remains unchanged in upstream and is an explicit difference from this adapter's numerical sampling setup.

The inputs are **unfiltered candidates used as numerical fixtures**. Identity normalization is used only for this witness. No corpus statistics are fitted, no optimizer is constructed, optimizer_updates=0, dataset_accepted=false and semantic_quality_evaluated=false. Passing this command does not accept the dataset, train a model, validate the complete dataloader, measure long-run stability, demonstrate physical execution, or establish full paper equivalence. Existing57D training stays paused. Future training still requires real tracking-filter evidence, accepted69D data/statistics/cache, integration validation and resource preflight.

An existing successful output is preserved and the command refuses to overwrite it. Failure traces are saved separately with UTC suffixes. Do not remove data or checkpoints to rerun this command; use a separately documented output name if another validation is required.
