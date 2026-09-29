#!/usr/bin/env bash
set -euo pipefail

# Export the fixed I1 kinematic correction as separate G1 references. This is
# not diffusion retraining and it never overwrites the official retarget files.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_export_corrected_refs.state"
mkdir -p "$MON"
exec 9>"$MON/r3_i1_export_corrected_refs.lock"
flock -n 9 || { echo 'already running' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
for SEED in 42 43 44; do
  STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed${SEED}"
  DEST="$ROOT/artifacts/i1_corrected_seed${SEED}"
  if [[ -e "$DEST" ]]; then
    echo "BLOCKED seed=$SEED export_dir_exists=$DEST" > "$STATE"
    exit 1
  fi
  echo "RUNNING seed=$SEED" > "$STATE"
  if ! timeout 900s "$OLD/.venv-retarget/bin/python" -u \
    "$ROOT/scripts/r3_i1_robotspace_ik_pilot.py" \
    --stage "$STAGE" --output "$ROOT/results/r3_i1_corrected_export_seed${SEED}.json" \
    --max-samples 100 --steps 80 --mode window_smooth \
    --max-joint-change-rad 1.0 --export-dir "$DEST" \
    > "$MON/r3_i1_export_corrected_seed${SEED}.log" 2>&1; then
    echo "FAILED seed=$SEED log=$MON/r3_i1_export_corrected_seed${SEED}.log" > "$STATE"
    exit 1
  fi
  case "$SEED" in 42|43) EXPECTED=64;; 44) EXPECTED=63;; esac
  COUNT="$(find "$DEST" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
  if [[ "$COUNT" -ne "$EXPECTED" ]]; then
    echo "FAILED seed=$SEED export_count=$COUNT expected=$EXPECTED" > "$STATE"
    exit 1
  fi
done
echo "COMPLETE seeds=42,43,44 outputs=191 at=$(date -u +%FT%TZ)" > "$STATE"
