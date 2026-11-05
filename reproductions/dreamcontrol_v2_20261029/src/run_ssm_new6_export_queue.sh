#!/usr/bin/env bash
set -u
cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/ssm_new6_20hz_20261101 || exit 1
ROOT=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
printf '%s\n' waiting_for_human_gpu > queue_status.txt
for attempt in $(seq 1 1080); do
  status=$(cat "$ROOT/ssm_new6_anchor_gpu_v2_20261101/queue_status.txt" 2>/dev/null || true)
  if [[ "$status" == audited ]]; then break; fi
  if [[ "$status" == solver_failed || "$status" == audit_failed || "$status" == blocked_omni_failed || "$status" == human_preparation_incomplete || "$status" == resource_wait_timeout || "$status" == wait_timeout ]]; then
    printf '%s\n' "blocked_human:$status" > queue_status.txt; exit 1
  fi
  sleep 10
done
if [[ "$status" != audited ]]; then printf '%s\n' wait_timeout > queue_status.txt; exit 1; fi
disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
if ! [[ "$disk" =~ ^[0-9]+$ ]] || (( disk < 8589934592 )); then printf '%s\n' insufficient_disk > queue_status.txt; exit 1; fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen
printf '%s\n' exporting > queue_status.txt
if ! nice -n 15 timeout 600 "$PY" -u export_ssm_new6_20hz.py > run.log 2>&1; then
  printf '%s\n' export_failed > queue_status.txt; exit 1
fi
printf '%s\n' auditing > queue_status.txt
if ! nice -n 15 timeout 600 "$PY" -u audit_ssm_new6_20hz.py > audit.log 2>&1; then
  printf '%s\n' audit_failed > queue_status.txt; exit 1
fi
printf '%s\n' audited > queue_status.txt
