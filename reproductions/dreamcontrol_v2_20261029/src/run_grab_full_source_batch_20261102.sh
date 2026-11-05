#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/grab_full_human_work_20261102" || exit 1
echo running > queue_status.txt
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 15 timeout 3600 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python reconstruct_grab_full_source_batch_20261102.py --selection selection.json --out "$ROOT/grab_full_human_source_batch_20261102" > run.log 2>&1
result=$?
if [ "$result" -eq 0 ]; then
  echo finished_not_audited >> queue_status.txt
else
  echo "stopped_exit_$result" >> queue_status.txt
fi
exit "$result"
