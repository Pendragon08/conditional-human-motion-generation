#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/bml_global_features_20261031
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
for item in S04_Expert S08_Novice; do
  for ((i=0;i<250;i++)); do
    if grep -q "^$item encoded$" export_status.txt 2>/dev/null; then
      if nice -n 15 "$PY" audit.py "$item" > "$item.audit.log" 2>&1; then
        printf '%s %s\n' "$item" audited >> feature_audit_status.txt
      else
        printf '%s %s\n' "$item" audit_failed >> feature_audit_status.txt
      fi
      break
    fi
    if grep -Eq "^$item (encode_failed|skipped_failed_upstream)$" export_status.txt 2>/dev/null; then
      printf '%s %s\n' "$item" skipped_failed_upstream >> feature_audit_status.txt
      break
    fi
    sleep 10
  done
done
