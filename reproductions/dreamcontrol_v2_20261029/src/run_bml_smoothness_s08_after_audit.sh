#!/usr/bin/env bash
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
cd "$ROOT/bml_joint_smoothness_features_20261031"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
for ((i=0;i<130;i++)); do
  if grep -q '^S08_Novice audited$' "$ROOT/bml_joint_smoothness_20261031/queue_status.txt" 2>/dev/null; then
    if nice -n 15 "$PY" export.py S08_Novice > S08_Novice.export.log 2>&1; then
      if nice -n 15 "$PY" audit.py S08_Novice > S08_Novice.audit.log 2>&1; then
        printf '%s\n' 'feature_audited' > S08_Novice.status
      else
        printf '%s\n' 'feature_audit_failed' > S08_Novice.status
      fi
    else
      printf '%s\n' 'feature_export_failed' > S08_Novice.status
    fi
    if ! nice -n 15 "$PY" compare.py S08_Novice > S08_Novice.comparison.log 2>&1; then
      printf '%s\n' 'comparison_failed' > S08_Novice.comparison_status
    fi
    exit 0
  fi
  if grep -Eq '^S08_Novice (solver_failed|audit_failed)$' "$ROOT/bml_joint_smoothness_20261031/queue_status.txt" 2>/dev/null; then
    printf '%s\n' 'upstream_failed' > S08_Novice.status
    exit 1
  fi
  sleep 10
done
printf '%s\n' 'upstream_timeout' > S08_Novice.status
exit 1
