#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
BASE="$ROOT/artifacts/i1_fixed_candidates"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_fixed_candidates_gpu1.state"
mkdir -p "$BASE" "$MON"

if [[ -s "$BASE/seed42/results.npy" || -s "$BASE/seed43/results.npy" || -s "$BASE/seed44/results.npy" ]]; then
  printf 'BLOCKED partial_output_exists; inspect before retry\n' > "$STATE"
  exit 1
fi

GPU_USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "$GPU_USED" =~ ^[0-9]+$ ]] || (( GPU_USED > 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$GPU_USED" > "$STATE"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$ROOT/third_party/DreamControl/TrajGen${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$ROOT/third_party/DreamControl/TrajGen"

printf 'RUNNING constraints=100 fixed_hint_seed=20260929 sampling_seeds=42,43,44 gpu=1 start=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for SEED in 42 43 44; do
  OUT="$BASE/seed$SEED"
  LOG="$MON/r3_i1_fixed_candidates_seed${SEED}.log"
  printf 'RUNNING seed=%s start=%s\n' "$SEED" "$(date -u +%FT%TZ)" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=30s 3600s \
    "$OLD/.venv-pbhc-training/bin/python" -u "$ROOT/scripts/r3_i1_generate_fixed_hints.py" \
      --r3_constraint_seed 20260929 --seed "$SEED" \
      --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt \
      --num_repetitions 1 --task kick --output_dir "$OUT" --batch_size 128 \
      > "$LOG" 2>&1
  CODE=$?
  set -e
  if (( CODE != 0 )) || [[ ! -s "$OUT/results.npy" ]]; then
    printf 'FAILED seed=%s exit=%s log=%s\n' "$SEED" "$CODE" "$LOG" > "$STATE"
    exit 1
  fi
  printf 'COMPLETED_SEED seed=%s output=%s\n' "$SEED" "$OUT/results.npy" > "$STATE"
done
printf 'COMPLETE seeds=42,43,44 constraints=100 at=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
