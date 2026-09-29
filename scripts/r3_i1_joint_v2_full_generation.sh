#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
BASE="$ROOT/artifacts/i1_joint_v2_candidates"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_joint_v2_full_generation.state"
mkdir -p "$MON" "$BASE" "$ROOT/results"
exec 9>"$MON/r3_i1_joint_v2_full_generation.lock"
flock -n 9 || { echo 'Full generation already active' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$TRAJ${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$TRAJ"
for seed in 42 43 44; do
  OUT="$BASE/seed$seed"
  BASELINE="$ROOT/artifacts/i1_fixed_candidates/seed$seed/results.npy"
  LOG="$MON/r3_i1_joint_v2_full_generation_seed$seed.log"
  [[ -s "$BASELINE" && ! -e "$OUT" ]] || {
    echo "BLOCKED seed=$seed baseline_missing_or_candidate_exists" > "$STATE"; exit 1;
  }
  echo "RUNNING seed=$seed count=100 gpu=1 no_extra_ik=true" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=30s 7200s \
    "$OLD/.venv-pbhc-training/bin/python" -u \
    "$ROOT/scripts/r3_i1_generate_joint_conditions.py" \
    --r3_constraint_seed 20260929 --r3_pilot_count 100 \
    --r3_condition_mode joint_v2 --seed "$seed" \
    --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt \
    --num_repetitions 1 --task kick --output_dir "$OUT" --batch_size 128 \
    > "$LOG" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )) || [[ ! -s "$OUT/results.npy" ]]; then
    echo "FAILED seed=$seed exit=$rc log=$LOG" > "$STATE"; exit 1
  fi
  "$OLD/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_i1_generation_joint_compare.py" \
    --base "$BASE" --arm joint_v2 --expected-count 100 \
    --baseline-file "$BASELINE" --candidate-file "$OUT/results.npy" \
    --output "$ROOT/results/r3_i1_joint_v2_human_seed$seed.json" \
    > "$MON/r3_i1_joint_v2_compare_seed$seed.log" 2>&1 || {
    echo "FAILED seed=$seed matched_input_or_human_audit" > "$STATE"; exit 1;
  }
done
echo 'COMPLETE seeds=42,43,44 count=100 no_extra_ik=true' > "$STATE"
