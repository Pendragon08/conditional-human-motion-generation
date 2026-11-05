#!/usr/bin/env bash
set -uC
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python
CATALOG_SHA=5d6bd3c42f70a622f46ada6b9f259bdf25b36daa6f57ba31254d55343139404e
printf '%s\n' started > queue_status.txt
for sample in 007410 011120 002941; do
  if ! nice -n 15 timeout 300 taskset -c 18 "$PY" reconstruct_ssm_fullrate_candidate.py --id "$sample" \
       --catalog-sha256 "$CATALOG_SHA" --output "$PWD/$sample" > "${sample}_reconstruct.log" 2>&1; then
    printf '%s\n' "reconstruct_failed:$sample" >> queue_status.txt
    exit 1
  fi
  if ! nice -n 15 timeout 120 taskset -c 18 "$PY" audit_ssm_fullrate_candidate.py --output "$PWD/$sample" > "${sample}_audit.log" 2>&1; then
    printf '%s\n' "audit_failed:$sample" >> queue_status.txt
    exit 1
  fi
  printf '%s\n' "audited:$sample" >> queue_status.txt
done
printf '%s\n' complete >> queue_status.txt
