#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/bml_joint_smoothness_20261031
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for item in S04_Expert S08_Novice; do
  printf '%s %s\n' "$item" started >> queue_status.txt
  if bash "$item/run.sh"; then
    if (cd "$item" && nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python ../audit.py > audit.log 2>&1); then
      printf '%s %s\n' "$item" audited >> queue_status.txt
    else
      printf '%s %s\n' "$item" audit_failed >> queue_status.txt
    fi
  else
    printf '%s %s\n' "$item" solver_failed >> queue_status.txt
  fi
done
