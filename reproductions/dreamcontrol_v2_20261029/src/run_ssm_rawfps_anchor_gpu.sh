#!/usr/bin/env bash
set -uC
cd "$(dirname "$0")"
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
printf '%s\n' started > queue_status.txt
free_mb=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
free_bytes=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
if ! [[ "$free_mb" =~ ^[0-9]+$ && "$free_bytes" =~ ^[0-9]+$ ]] || (( free_mb < 10240 || free_bytes < 2147483648 )); then
  printf '%s\n' insufficient_resources >> queue_status.txt
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export PYTHONPATH="$ROOT/jax_cuda12_plugin_0_6_2_20261031:$ROOT/pyroki_cpu_deps_20261030/extra-packages:$ROOT/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils"
if ! nice -n 15 timeout 3600 taskset -c 18 "$PY" pyroki_ssm_rawfps_anchor_gpu.py > solver.log 2>&1; then
  printf '%s\n' solver_failed >> queue_status.txt
  exit 1
fi
if ! CUDA_VISIBLE_DEVICES= nice -n 15 timeout 120 "$PY" audit_ssm_rawfps_gpu_batch.py > audit.log 2>&1; then
  printf '%s\n' audit_failed >> queue_status.txt
  exit 1
fi
printf '%s\n' audited >> queue_status.txt
if ! CUDA_VISIBLE_DEVICES= nice -n 15 timeout 120 "$PY" compare_ssm_anchor_batch.py > comparison.log 2>&1; then
  printf '%s\n' comparison_failed >> queue_status.txt
  exit 1
fi
printf '%s\n' compared >> queue_status.txt
