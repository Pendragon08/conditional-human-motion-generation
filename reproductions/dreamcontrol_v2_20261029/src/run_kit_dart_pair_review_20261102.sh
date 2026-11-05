#!/usr/bin/env bash
set -uC
cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/kit_dart_hypothesis_work_20261102 || exit 1
echo running > pair_render_status.txt
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 15 timeout 1800 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python render_kit_dart_pair_review_20261102.py > pair_render.log 2>&1
result=$?
if [ "$result" -eq 0 ]; then
  echo generated_not_reviewed >> pair_render_status.txt
else
  echo "stopped_exit_$result" >> pair_render_status.txt
fi
exit "$result"
