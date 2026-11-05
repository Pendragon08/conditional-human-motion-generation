#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/accad_013267_fullrate_candidate_20261031"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
for ((i=0;i<250;i++)); do
  if grep -q '^audited$' "$ROOT/accad_013267_pyroki_fullrate_20261031/queue_status.txt" 2>/dev/null; then
    if nice -n 15 "$PY" export.py > export.log 2>&1; then
      if nice -n 15 "$PY" audit.py > audit.log 2>&1; then
        printf '%s\n' 'audited' > candidate_status.txt
      else
        printf '%s\n' 'audit_failed' > candidate_status.txt
      fi
    else
      printf '%s\n' 'export_failed' > candidate_status.txt
    fi
    exit 0
  fi
  if grep -Eq '^(solver_failed|audit_failed)$' "$ROOT/accad_013267_pyroki_fullrate_20261031/queue_status.txt" 2>/dev/null; then
    printf '%s\n' 'upstream_failed' > candidate_status.txt
    exit 1
  fi
  sleep 10
done
printf '%s\n' 'upstream_timeout' > candidate_status.txt
exit 1
