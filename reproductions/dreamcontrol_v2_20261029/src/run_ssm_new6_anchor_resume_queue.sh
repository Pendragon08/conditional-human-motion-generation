#!/usr/bin/env bash
set -u
cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/ssm_new6_anchor_resume_20261101 || exit 1
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
IDS='000162 004028 006749 007129 008894 009466'
printf '%s\n' waiting_for_resources > queue_status.txt
ready=0
for attempt in $(seq 1 240); do
  line=$(nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits)
  IFS=, read -r util free total <<< "$line"
  util=${util//[[:space:]]/};free=${free//[[:space:]]/};total=${total//[[:space:]]/}
  disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
  if [[ "$util" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$disk" =~ ^[0-9]+$ ]] && \
     (( util <= 10 && free >= total / 10 + 2048 && disk >= 8589934592 )); then ready=1; break; fi
  sleep 30
done
if (( ready == 0 )); then printf '%s\n' resource_wait_timeout > queue_status.txt; exit 1; fi
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export PYTHONPATH="$ROOT/jax_cuda12_plugin_0_6_2_20261031:$ROOT/pyroki_cpu_deps_20261030/extra-packages:$ROOT/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils"
printf '%s\n' running > queue_status.txt
if ! nice -n 15 timeout 7200 taskset -c 18 "$PY" pyroki_ssm_new6_anchor_resume_gpu.py \
      --human-root "$ROOT/ssm_new6_human_20261101" --samples $IDS > solver.log 2>&1; then
  printf '%s\n' solver_failed > queue_status.txt; exit 1
fi
printf '%s\n' auditing > queue_status.txt
if ! CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 15 timeout 300 "$PY" audit_ssm_rawfps_gpu_batch.py \
      --human-root "$ROOT/ssm_new6_human_20261101" --samples $IDS > audit.log 2>&1; then
  printf '%s\n' audit_failed > queue_status.txt; exit 1
fi
printf '%s\n' audited > queue_status.txt
