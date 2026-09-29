#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
BASE="$ROOT/artifacts/r3_i1_generation_joint_pilot"
STATE="$MON/r3_i1_generation_joint_pilot.state"
mkdir -p "$MON" "$BASE"
exec 9>"$MON/r3_i1_generation_joint_pilot.lock"
flock -n 9 || { echo 'Another joint-conditions pilot is active' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$TRAJ${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$TRAJ"
for arm in baseline joint; do
  OUT="$BASE/$arm"
  LOG="$MON/r3_i1_generation_joint_pilot_${arm}.log"
  [[ ! -e "$OUT" ]] || { echo "BLOCKED existing_output=$OUT" > "$STATE"; exit 1; }
  echo "RUNNING arm=$arm gpu=1 count=8 seed=42 fixed_constraints=20260929" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=30s 7200s \
    "$OLD/.venv-pbhc-training/bin/python" -u \
    "$ROOT/scripts/r3_i1_generate_joint_conditions.py" \
    --r3_constraint_seed 20260929 --r3_pilot_count 8 \
    --r3_condition_mode "$arm" --seed 42 \
    --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt \
    --num_repetitions 1 --task kick --output_dir "$OUT" --batch_size 8 \
    > "$LOG" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )) || [[ ! -s "$OUT/results.npy" ]]; then
    echo "FAILED arm=$arm exit=$rc log=$LOG" > "$STATE"
    exit 1
  fi
done
echo "COMPLETE arms=baseline,joint count=8 seed=42" > "$STATE"
