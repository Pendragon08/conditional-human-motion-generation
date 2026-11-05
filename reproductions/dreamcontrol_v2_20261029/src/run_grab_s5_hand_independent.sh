#!/usr/bin/env bash
set -euC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/grab_s5_hand_independent_20261030"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
ionice -c 3 nice -n 15 taskset -c 19 "$PY" run.py > solver.log 2>&1
