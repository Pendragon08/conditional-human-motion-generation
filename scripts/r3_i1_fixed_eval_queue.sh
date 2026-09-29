#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_fixed_eval_queue.state"
mkdir -p "$MON" "$ROOT/results"
printf 'WAITING_FOR_RETARGET at=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 240); do
  if grep -q '^COMPLETE matched_seeds=42,43,44' "$MON/r3_i1_fixed_retarget_queue.state" 2>/dev/null; then
    break
  fi
  if grep -q '^FAILED\|^BLOCKED' "$MON/r3_i1_fixed_retarget_queue.state" 2>/dev/null; then
    printf 'BLOCKED retarget_failed\n' > "$STATE"
    exit 1
  fi
  sleep 30
done
if ! grep -q '^COMPLETE matched_seeds=42,43,44' "$MON/r3_i1_fixed_retarget_queue.state" 2>/dev/null; then
  printf 'BLOCKED retarget_wait_timeout\n' > "$STATE"
  exit 1
fi

export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
for SEED in 42 43 44; do
  STAGE="$TRAJ/sample_i1_fixed_seed${SEED}"
  printf 'RUNNING_AUDIT seed=%s\n' "$SEED" > "$STATE"
  timeout 300s "$OLD/.venv-retarget/bin/python" -u "$ROOT/scripts/r3_i1_constraint_transfer_audit.py" \
    --stage "$STAGE" --output "$ROOT/results/r3_i1_fixed_seed${SEED}_audit.json" \
    > "$MON/r3_i1_fixed_seed${SEED}_audit.log" 2>&1 || {
    printf 'FAILED_AUDIT seed=%s\n' "$SEED" > "$STATE"; exit 1;
  }
  printf 'RUNNING_IK seed=%s\n' "$SEED" > "$STATE"
  timeout 600s "$OLD/.venv-retarget/bin/python" -u "$ROOT/scripts/r3_i1_robotspace_ik_pilot.py" \
    --stage "$STAGE" --output "$ROOT/results/r3_i1_fixed_seed${SEED}_ik.json" \
    --max-samples 100 --steps 80 --mode window_smooth --max-joint-change-rad 1.0 \
    > "$MON/r3_i1_fixed_seed${SEED}_ik.log" 2>&1 || {
    printf 'FAILED_IK seed=%s\n' "$SEED" > "$STATE"; exit 1;
  }
done
python3 "$ROOT/scripts/r3_i1_multi_seed_summary.py" --results "$ROOT/results" \
  --output "$ROOT/results/r3_i1_fixed_three_seed_summary.json" \
  > "$MON/r3_i1_fixed_three_seed_summary.log" 2>&1 || {
  printf 'FAILED_SUMMARY\n' > "$STATE"; exit 1;
}
printf 'COMPLETE at=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
