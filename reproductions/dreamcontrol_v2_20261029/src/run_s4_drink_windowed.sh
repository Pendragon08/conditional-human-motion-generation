#!/usr/bin/env bash
set -euC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/pyroki_grab_s4_drink_windowed_20261030"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
export PYTHONPATH="$ROOT/pyroki_cpu_deps_20261030/extra-packages:$ROOT/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils"
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
ionice -c 3 nice -n 15 taskset -c 19 "$PY" pyroki_grab_windowed_pilot.py > solver.log 2>&1
ionice -c 3 nice -n 15 taskset -c 19 "$PY" audit_pyroki_windowed_pilot.py > audit.log 2>&1
