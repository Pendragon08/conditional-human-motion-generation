#!/usr/bin/env bash
set -euC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/pyroki_ssm_000008_human_20261031"
while kill -0 768709 2>/dev/null; do sleep 10; done
if [ ! -f report.json ]; then
  printf '%s\n' 'Solver ended without report; retain logs and partial windows.' > audit_skipped.txt
  exit 1
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ionice -c 3 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit.py > audit.log 2>&1
