#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/accad_013267_anchor_fullrate_20261031
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s\n' 'started' > queue_status.txt
if bash run.sh; then
  if nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit.py > audit.log 2>&1; then
    printf '%s\n' 'audited' >> queue_status.txt
  else
    printf '%s\n' 'audit_failed' >> queue_status.txt
  fi
else
  printf '%s\n' 'solver_failed' >> queue_status.txt
fi
