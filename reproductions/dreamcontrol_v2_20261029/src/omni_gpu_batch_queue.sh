#!/bin/bash
set -u
cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/omni_gpu_remaining_20261101 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.:../gradient_accumulation_v2_20261101/src:../training_loop_20261030:../resume_state_audit_20261031:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
printf 'running\n' > queue_status.txt
if ! CUDA_VISIBLE_DEVICES=1 "$PY" -u omni_batch_benchmark.py --remaining-from ../omni_gpu_batch24_20261101/selection.json > run.log 2>&1; then
  printf 'processing_failed\n' > queue_status.txt
  exit 1
fi
printf 'auditing\n' > queue_status.txt
if ! CUDA_VISIBLE_DEVICES= "$PY" -u audit_omni_gpu_batch.py > audit.log 2>&1; then
  printf 'audit_failed\n' > queue_status.txt
  exit 1
fi
printf 'audited\n' > queue_status.txt
