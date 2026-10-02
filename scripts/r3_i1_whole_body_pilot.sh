#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_whole_body_pilot.state"
LOG="$MON/r3_i1_whole_body_pilot.log"
OUT="$ROOT/results/r3_i1_whole_body_pilot_seed42.json"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_whole_body_pilot.lock"
flock -n 9 || { echo 'Whole-body pilot already running' >&2; exit 1; }
[[ -s "$STAGE/Kick_sim/results.npy" ]] || { echo 'BLOCKED missing_generated_motion' > "$STATE"; exit 1; }
[[ ! -f "$OUT" ]] || { echo 'BLOCKED output_already_exists' > "$STATE"; exit 1; }
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
echo "RUNNING pilot=8 seed=42 steps=120 raw_retarget_unchanged=true" > "$STATE"
set +e
timeout --signal=TERM --kill-after=20s 3600s "$OLD/.venv-retarget/bin/python" -u \
  "$ROOT/scripts/r3_i1_whole_body_refine.py" \
  --stage "$STAGE" --output "$OUT" --max-samples 8 --steps 120 --device cpu \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || [[ ! -s "$OUT" ]]; then
  echo "FAILED exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE output=$OUT log=$LOG" > "$STATE"
