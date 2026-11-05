#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/grab_full_human_work_v3_20261102" || exit 1
echo running > queue_status.txt
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
nice -n 15 timeout 3600 "$PYTHON" reconstruct_grab_full_source_batch_v2_20261102.py --selection selection.json --out "$ROOT/grab_full_human_source_batch_v3_20261102" > run.log 2>&1
result=$?
if [ "$result" -ne 0 ]; then
  echo "reconstruction_stopped_exit_$result" >> queue_status.txt
  exit "$result"
fi
echo finished_not_audited >> queue_status.txt
nice -n 15 timeout 1800 "$PYTHON" audit_grab_full_source_batch_v2_20261102.py --selection selection.json --batch "$ROOT/grab_full_human_source_batch_v3_20261102" --output audit.json > audit.log 2>&1
result=$?
if [ "$result" -eq 0 ]; then
  echo audited_check_results >> queue_status.txt
else
  echo "audit_stopped_exit_$result" >> queue_status.txt
fi
exit "$result"
