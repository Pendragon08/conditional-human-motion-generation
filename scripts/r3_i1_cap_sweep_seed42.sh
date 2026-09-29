#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_cap_sweep_seed42.state"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42"
for ITEM in '025:0.25' '050:0.50'; do
  TAG="${ITEM%%:*}"
  CAP="${ITEM#*:}"
  DEST="$ROOT/artifacts/i1_corrected_seed42_cap${TAG}"
  OUT="$ROOT/results/r3_i1_corrected_export_seed42_cap${TAG}.json"
  [[ ! -e "$DEST" && ! -e "$OUT" ]] || { echo "BLOCKED existing_output=$TAG" > "$STATE"; exit 1; }
  echo "RUNNING cap=$CAP seed=42" > "$STATE"
  if ! timeout 900s "$OLD/.venv-retarget/bin/python" -u \
    "$ROOT/scripts/r3_i1_robotspace_ik_pilot.py" \
    --stage "$STAGE" --output "$OUT" --max-samples 100 \
    --steps 80 --mode window_smooth --max-joint-change-rad "$CAP" \
    --export-dir "$DEST" > "$MON/r3_i1_cap_sweep_seed42_cap${TAG}.log" 2>&1; then
    echo "FAILED cap=$CAP" > "$STATE"; exit 1
  fi
  [[ "$(find "$DEST" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 64 ]] || {
    echo "FAILED export_count cap=$CAP" > "$STATE"; exit 1;
  }
done
echo "COMPLETE caps=0.25,0.50 seed=42 at=$(date -u +%FT%TZ)" > "$STATE"
