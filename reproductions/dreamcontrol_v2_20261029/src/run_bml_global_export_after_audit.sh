#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
WORK="$ROOT/bml_global_features_20261031"
QUEUE="$ROOT/bml_global_candidates_20261031"
cd "$WORK"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$WORK:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen"
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
for item in S04_Expert S08_Novice; do
  for ((i=0;i<245;i++)); do
    if grep -q "^$item audited$" "$QUEUE/queue_status.txt" 2>/dev/null; then
      if nice -n 15 "$PY" export.py "$item" > "$item.export.log" 2>&1; then
        printf '%s %s\n' "$item" encoded >> export_status.txt
      else
        printf '%s %s\n' "$item" encode_failed >> export_status.txt
      fi
      break
    fi
    if grep -Eq "^$item (solver_failed|audit_failed)$" "$QUEUE/queue_status.txt" 2>/dev/null; then
      printf '%s %s\n' "$item" skipped_failed_upstream >> export_status.txt
      break
    fi
    sleep 10
  done
done
