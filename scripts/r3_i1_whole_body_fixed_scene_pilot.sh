#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_whole_body_fixed_scene_pilot.state"
LOG="$MON/r3_i1_whole_body_fixed_scene_pilot.log"
OUT="$ROOT/results/r3_i1_whole_body_fixed_scene_pilot_seed42.json"
EXPORT="$ROOT/artifacts/r3_i1_whole_body_fixed_scene_pilot_seed42"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_whole_body_fixed_scene_pilot.lock"
flock -n 9 || { echo 'Fixed-scene pilot already running' >&2; exit 1; }
[[ ! -e "$OUT" && ! -e "$EXPORT" ]] || { echo 'BLOCKED output_exists' > "$STATE"; exit 1; }
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
echo 'RUNNING seed=42 pilot=8 fixed_scene_target=true old_single_leg_ik=false' > "$STATE"
set +e
timeout --signal=TERM --kill-after=20s 3600s "$OLD/.venv-retarget/bin/python" -u \
  "$ROOT/scripts/r3_i1_whole_body_refine.py" \
  --stage "$STAGE" --output "$OUT" --export-dir "$EXPORT" \
  --max-samples 8 --steps 200 --device cpu \
  --max-joint-change-rad 0.10 --max-root-xy-change-m 0.02 \
  --max-root-z-change-m 0.005 --velocity-weight 20 \
  --acceleration-weight 30 --peak-weight 1.0 > "$LOG" 2>&1
rc=$?
set -e
actual="$(find "$EXPORT" -maxdepth 1 -name '*.pkl' -type f 2>/dev/null | wc -l)"
if (( rc != 0 )) || [[ ! -s "$OUT" ]] || (( actual != 8 )); then
  echo "FAILED exit=$rc exported=$actual log=$LOG" > "$STATE"; exit 1
fi
echo "COMPLETE output=$OUT exported=8 log=$LOG" > "$STATE"
