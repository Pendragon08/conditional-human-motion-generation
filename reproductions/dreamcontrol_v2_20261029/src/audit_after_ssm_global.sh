#!/usr/bin/env bash
set -euC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/pyroki_ssm_000008_global_20261031"
while kill -0 774415 2>/dev/null; do sleep 10; done
if [ ! -f report.json ]; then
  printf '%s\n' 'Global solve ended without report; retain logs and partial output.' > audit_skipped.txt
  exit 1
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ionice -c 3 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit.py > audit.log 2>&1
