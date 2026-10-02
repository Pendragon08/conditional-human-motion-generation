#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_whole_body_conservative_three_seed.state"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_whole_body_conservative_three_seed.lock"
flock -n 9 || { echo 'Three-seed refinement already running' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
for seed in 42 43 44; do
  STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed$seed"
  OUT="$ROOT/results/r3_i1_whole_body_conservative_seed$seed.json"
  EXPORT="$ROOT/artifacts/r3_i1_whole_body_conservative_seed$seed"
  LOG="$MON/r3_i1_whole_body_conservative_seed$seed.log"
  [[ -s "$STAGE/Kick_sim/results.npy" ]] || { echo "BLOCKED seed=$seed missing_source" > "$STATE"; exit 1; }
  [[ ! -e "$OUT" && ! -e "$EXPORT" ]] || { echo "BLOCKED seed=$seed output_exists" > "$STATE"; exit 1; }
  expected="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
  echo "RUNNING seed=$seed raw_count=$expected fixed_hyperparameters=true" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=20s 3600s "$OLD/.venv-retarget/bin/python" -u \
    "$ROOT/scripts/r3_i1_whole_body_refine.py" \
    --stage "$STAGE" --output "$OUT" --export-dir "$EXPORT" \
    --max-samples 100 --steps 200 --device cpu \
    --max-joint-change-rad 0.10 --max-root-xy-change-m 0.02 \
    --max-root-z-change-m 0.005 --velocity-weight 20 \
    --acceleration-weight 30 --peak-weight 1.0 > "$LOG" 2>&1
  rc=$?
  set -e
  actual="$(find "$EXPORT" -maxdepth 1 -name '*.pkl' -type f 2>/dev/null | wc -l)"
  if (( rc != 0 )) || [[ ! -s "$OUT" ]] || (( actual != expected )); then
    echo "FAILED seed=$seed exit=$rc exported=$actual expected=$expected log=$LOG" > "$STATE"
    exit 1
  fi
done
echo 'COMPLETE seeds=42,43,44 fixed_hyperparameters=true' > "$STATE"
