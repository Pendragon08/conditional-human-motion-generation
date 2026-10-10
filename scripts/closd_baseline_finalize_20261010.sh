#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
OUT="$ROOT/results/closd_baseline_20261010"
PY="$RUNTIME/.venv-pbhc-training/bin/python"
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
exec 9>"$OUT/finalize.lock"
flock -n 9 || exit 1
cd "$ROOT/third_party/CLoSD"
for attempt in $(seq 1 480); do
  python3 "$ROOT/scripts/closd_baseline_report_20261010.py" --root "$OUT" > "$OUT/report_update.log"
  if grep -q '^COMPLETE' "$OUT/pipeline.state"; then
    "$PY" "$ROOT/scripts/closd_baseline_preview_20261010.py" --root "$OUT" \
      --asset-root "$RUNTIME/cache/closd_hf/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b" \
      > "$OUT/preview.log" 2>&1
    python3 "$ROOT/scripts/closd_baseline_report_20261010.py" --root "$OUT" > "$OUT/report_update.log"
    echo COMPLETE_REPORT_AND_PREVIEWS > "$OUT/finalize.state"
    exit 0
  fi
  if grep -q '^FAILED' "$OUT/pipeline.state"; then
    echo PIPELINE_FAILED_REPORT_SAVED > "$OUT/finalize.state"
    exit 1
  fi
  sleep 60
done
echo TIMEOUT_WAITING_FOR_PIPELINE > "$OUT/finalize.state"
exit 1
