#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
BASE="$ROOT/artifacts/i1_fixed_candidates"
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_fixed_retarget_queue.state"
mkdir -p "$MON" "$ROOT/results"
printf 'WAITING_FOR_GENERATION at=%s\n' "$(date -u +%FT%TZ)" > "$STATE"

for _ in $(seq 1 240); do
  if grep -q '^COMPLETE seeds=42,43,44' "$MON/r3_i1_fixed_candidates_gpu1.state" 2>/dev/null; then
    break
  fi
  if grep -q '^FAILED\|^BLOCKED' "$MON/r3_i1_fixed_candidates_gpu1.state" 2>/dev/null; then
    printf 'BLOCKED generation_failed\n' > "$STATE"
    exit 1
  fi
  sleep 30
done
if ! grep -q '^COMPLETE seeds=42,43,44' "$MON/r3_i1_fixed_candidates_gpu1.state" 2>/dev/null; then
  printf 'BLOCKED generation_wait_timeout\n' > "$STATE"
  exit 1
fi

if ! "$OLD/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_i1_verify_fixed_hints.py" \
  --base "$BASE" --output "$ROOT/results/r3_i1_fixed_hints_manifest.json" \
  > "$MON/r3_i1_fixed_hints_verify.log" 2>&1; then
  printf 'BLOCKED mismatched_or_invalid_conditions\n' > "$STATE"
  exit 1
fi

for SEED in 42 43 44; do
  STAGE="$TRAJ/sample_i1_fixed_seed${SEED}"
  if [[ -e "$STAGE" ]]; then
    printf 'BLOCKED stage_exists=%s\n' "$STAGE" > "$STATE"
    exit 1
  fi
  mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1"
  cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
  cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
  ln -s "$BASE/seed${SEED}/results.npy" "$STAGE/Kick_sim/results.npy"
  export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  printf 'RUNNING_RETARGET seed=%s at=%s\n' "$SEED" "$(date -u +%FT%TZ)" > "$STATE"
  set +e
  (cd "$STAGE/Kick_sim" && timeout --signal=TERM --kill-after=30s 1800s \
      "$OLD/.venv-retarget/bin/python" -u retarget.py) \
      > "$MON/r3_i1_retarget_seed${SEED}.log" 2>&1
  CODE=$?
  set -e
  COUNT=$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)
  if (( CODE != 0 || COUNT < 1 )); then
    printf 'FAILED_RETARGET seed=%s exit=%s count=%s\n' "$SEED" "$CODE" "$COUNT" > "$STATE"
    exit 1
  fi
  printf 'COMPLETE_RETARGET seed=%s count=%s\n' "$SEED" "$COUNT" > "$STATE"
done
printf 'COMPLETE matched_seeds=42,43,44 at=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
