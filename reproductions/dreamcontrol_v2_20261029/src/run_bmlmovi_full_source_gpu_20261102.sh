#!/usr/bin/env bash
# Isolated declared-source-rate BMLmovi preview, not a training job.
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/bmlmovi_full_robot_work_20261102" || exit 1
line=$(nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits) || exit 1
IFS=, read -r util free total <<< "$line"
util=${util//[[:space:]]/}; free=${free//[[:space:]]/}; total=${total//[[:space:]]/}
disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
if ! [[ "$util" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$disk" =~ ^[0-9]+$ ]]; then exit 1; fi
if (( util > 10 || free < total/20+2048 || disk < 8589934592+134217728 )); then echo resource_guard_stopped; exit 1; fi
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.05
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export PYTHONPATH="$ROOT/jax_cuda12_plugin_0_6_2_20261031:$ROOT/pyroki_cpu_deps_20261030/extra-packages:$ROOT/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils"
printf '%s\n' running > queue_status.txt
if nice -n 15 timeout 7200 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python retarget_bmlmovi_full_source_20261102.py --out robot_windows --samples 010639 012167 005851 012145 000433 011782 008475 011698 005665 002961 012818 000466 007552 000763 012221 009104 004016 006275 005475 010261 > solver.log 2>&1; then
    printf '%s\n' solved_not_audited >> queue_status.txt
else
    code=$?
    printf 'solver_stopped exit=%s\n' "$code" >> queue_status.txt
    exit "$code"
fi
