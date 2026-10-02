#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42"
RAW="$ROOT/artifacts/r3_i1_whole_body_raw8_seed42"
REFINED="$ROOT/artifacts/r3_i1_whole_body_pilot_seed42"
CKPT=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt
STATE="$MON/r3_i1_whole_body_paired_pilot.state"
mkdir -p "$MON" "$ROOT/results" "$RAW"
exec 9>"$MON/r3_i1_whole_body_paired_pilot.lock"
flock -n 9 || { echo 'Whole-body paired pilot already running' >&2; exit 1; }
[[ -s "$CKPT" ]] || { echo 'BLOCKED missing_fixed_tracker' > "$STATE"; exit 1; }
[[ "$(find -L "$REFINED" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 8 ]] || {
  echo 'BLOCKED refined_reference_count' > "$STATE"; exit 1;
}
for refined in "$REFINED"/*.pkl; do
  name="$(basename "$refined")"
  source="$STAGE/Kick_sim1/$name"
  [[ -s "$source" ]] || { echo "BLOCKED missing_raw=$name" > "$STATE"; exit 1; }
  if [[ ! -e "$RAW/$name" ]]; then ln -s "$source" "$RAW/$name"; fi
done
[[ "$(find -L "$RAW" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 8 ]] || {
  echo 'BLOCKED raw_reference_count' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
for arm in raw refined; do
  if [[ "$arm" == raw ]]; then export DREAMCONTROL_KICK_REF_PATH="$RAW";
  else export DREAMCONTROL_KICK_REF_PATH="$REFINED"; fi
  OUT="$ROOT/results/r3_i1_whole_body_pilot_${arm}_fixed_goal.json"
  LOG="$MON/r3_i1_whole_body_pilot_${arm}_fixed_goal.log"
  [[ ! -e "$OUT" ]] || { echo "BLOCKED output_exists=$OUT" > "$STATE"; exit 1; }
  echo "RUNNING arm=$arm envs=256 frames=500 seed=42 goal=frozen_raw_reference tracker=fixed" > "$STATE"
  cd "$TRAIN"
  set +e
  timeout --signal=TERM --kill-after=45s 3600s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
    --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
    --video_length=500 --id="r3_i1_whole_body_${arm}" \
    --freeze-task-goal-from-dir="$RAW" --metrics-out="$OUT" \
    > "$LOG" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )) || [[ ! -s "$OUT" ]]; then
    echo "FAILED arm=$arm exit=$rc log=$LOG" > "$STATE"; exit 1
  fi
done
echo "COMPLETE arms=raw,refined same_tracker=true same_initial_motion_ids=true" > "$STATE"
