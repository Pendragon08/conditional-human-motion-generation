# ReactiveBFM isolated compatibility runtime

Spec: `configs/env_spec.json`, canonical hash `5d436660`. Server: ouyanghao@10.147.17.32, SSH key C:/Users/Pende/.ssh/id_ed25519. Existing jobs must continue untouched. The user authorized a bounded shared-GPU verification. Python3.10.21/Torch2.5.1+cu121 differs from the upstream recommendation Python3.12/Torch2.11+cu128; this is a compatibility pilot. No shared package was changed.

Run the following command verbatim exactly once on the server. Its output file must be absent first. It uses shared GPU1 with a 25% PyTorch memory allocation cap (about12GiB), CPU4, seed42, the pinned official512x16 DiT and real Kungfu training data. It performs a CUDA kernel and official model forward/backward, without an optimizer. Do not run another GPU witness or training process for this project concurrently.

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code/scripts/witness_runtime.py --runtime-root /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021 --output /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/witness_document_v1.json
```

Success requires exit0 and final status `OFFICIAL_DIT_KERNEL_AND_BACKWARD_PASSED`, prediction_shape `[1,36,1,50]`, nonzero finite gradients, optimizer_updates0, 226 source hashes verified, official source commit `5a79d03cab824eb5917271ba720294e8a1b9d374`, correct Kungfu SHA256 and matching pinned package versions. Expected trainable parameter count107045924; GPU name NVIDIA RTX A6000. The value of the diagnostic finite loss is not a quality result.

Preserve exact command/stdout/stderr/receipt, start/end UTC, exit code, source and document before/after hashes. Report all divergences without changing code, packages, data or docs. Do not delete or upload motion arrays. This validates the selected runtime only; it is not generator training, MuJoCo control, paper metrics, target-reaching or hardware validation.
