#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIMIT="${1:-2}"
SEED="${2:-10}"
SUFFIX="n${LIMIT}"
if [[ "$SEED" != 10 ]]; then SUFFIX="${SUFFIX}_seed${SEED}"; fi
OUT="$ROOT/artifacts/dreamcontrol-r3/generated/kick_guided_n${LIMIT}_seed${SEED}"
STATE="$ROOT/artifacts/monitoring/r3_dreamcontrol_generate_${SUFFIX}.state"
LOG="$ROOT/artifacts/monitoring/r3_dreamcontrol_generate_${SUFFIX}.log"
mkdir -p "$(dirname "$STATE")" "$(dirname "$OUT")"
if [[ -e "$OUT/results.npy" ]]; then
  printf 'ALREADY_COMPLETE output=%s\n' "$OUT/results.npy" > "$STATE"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  printf 'BLOCKED output_dir_exists=%s\n' "$OUT" > "$STATE"
  exit 1
fi
used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu0_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLIP:$ROOT/third_party/DreamControl/TrajGen${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$ROOT/third_party/DreamControl/TrajGen"
printf 'RUNNING started=%s limit=%s seed=%s gpu=0\n' "$(date -u +%FT%TZ)" "$LIMIT" "$SEED" > "$STATE"
timeout --signal=TERM --kill-after=30s 1800s \
  "$ROOT/.venv-pbhc-training/bin/python" -u "$ROOT/scripts/r3_dreamcontrol_generate_safe.py" \
    --r3_limit "$LIMIT" --seed "$SEED" \
    --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt \
    --num_repetitions 1 --task kick --output_dir "$OUT" --batch_size 128 \
    > "$LOG" 2>&1
result=$?
if (( result != 0 )) || [[ ! -s "$OUT/results.npy" ]]; then
  printf 'BLOCKED exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'FINISHED_NEEDS_VALIDATION output=%s log=%s\n' "$OUT/results.npy" "$LOG" > "$STATE"
