#!/usr/bin/env bash
# Run only after all windows finish; leave every input and failed output intact.
set -uC
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
WORK="$ROOT/grab_robot_batch_v2_work_20261103"
OUT="$ROOT/grab_robot_batch_v2_263_preview_20261103"
cd "$WORK" || exit 1
if [ "$(tail -n 1 queue_status.txt)" != solved_not_audited ]; then exit 2; fi
if [ -e audit_windows.json ] || [ -e "$OUT" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/ssm_new6_20hz_resume_20261101:$ROOT/pyroki_window_blend_v2_20261030:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen"
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
echo running > finalize_status.txt
if ! nice -n 15 timeout 3600 "$PY" audit_grab_robot_batch_v2_20261103.py --output audit_windows.json > audit_windows.log 2>&1; then
  echo window_audit_failed >> finalize_status.txt; exit 4
fi
echo windows_audited >> finalize_status.txt
if ! nice -n 15 timeout 3600 "$PY" export_grab_robot_batch_v2_20261103.py --out "$OUT" > export.log 2>&1; then
  echo export_failed >> finalize_status.txt; exit 5
fi
echo exported_not_audited >> finalize_status.txt
if ! nice -n 15 timeout 3600 "$PY" audit_grab_robot_batch_263_v2_20261103.py > audit_export.log 2>&1; then
  echo export_audit_failed >> finalize_status.txt; exit 6
fi
echo audited_check_cases >> finalize_status.txt
