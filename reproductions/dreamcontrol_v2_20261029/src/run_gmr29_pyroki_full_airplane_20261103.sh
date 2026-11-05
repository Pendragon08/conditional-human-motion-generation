#!/usr/bin/env bash
set -euC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT"
[[ ! -e gmr29_pyroki_full_airplane_20261103 && ! -e gmr29_pyroki_full_airplane_20261103.log ]] || exit 1
line=$(nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits)
IFS=, read -r util free total <<< "$line"
util=${util//[[:space:]]/}; free=${free//[[:space:]]/}; total=${total//[[:space:]]/}
disk=$(df -B1 --output=avail /data | tail -1 | tr -d '[:space:]')
if (( util > 10 || free < total/20+2048 || disk < 8589934592+268435456 )); then echo resource_guard_stopped; exit 1; fi
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.05
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export PYTHONPATH="$ROOT/jax_cuda12_plugin_0_6_2_20261031:$ROOT/pyroki_cpu_deps_20261030/extra-packages:$ROOT/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils"
set +e
nice -n 15 timeout 3600 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python gmr29_pyroki_full_benchmark_20261103.py > gmr29_pyroki_full_airplane_20261103.log 2>&1
code=$?
printf 'exit=%s\n' "$code" > gmr29_pyroki_full_airplane_20261103.status
exit "$code"
