#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
RAW="$ROOT/artifacts/r3_i1_whole_body_raw8_seed42"
REF="$ROOT/artifacts/r3_i1_whole_body_pilot_seed42_conservative"
CKPT=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt
STATE="$MON/r3_i1_whole_body_conservative_eval.state"
OUT="$ROOT/results/r3_i1_whole_body_pilot_conservative_fixed_goal.json"
LOG="$MON/r3_i1_whole_body_pilot_conservative_fixed_goal.log"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_whole_body_conservative_eval.lock"
flock -n 9 || { echo 'Conservative evaluation already running' >&2; exit 1; }
[[ -s "$CKPT" && ! -e "$OUT" ]] || { echo 'BLOCKED checkpoint_missing_or_output_exists' > "$STATE"; exit 1; }
for directory in "$RAW" "$REF"; do
  [[ "$(find -L "$directory" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 8 ]] || {
    echo "BLOCKED reference_count=$directory" > "$STATE"; exit 1;
  }
done
export CUDA_VISIBLE_DEVICES=1 DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo 'RUNNING arm=conservative envs=256 frames=500 seed=42 goal=frozen_raw_reference tracker=fixed' > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
  --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
  --video_length=500 --id=r3_i1_whole_body_conservative \
  --freeze-task-goal-from-dir="$RAW" --metrics-out="$OUT" \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || [[ ! -s "$OUT" ]]; then
  echo "FAILED exit=$rc log=$LOG" > "$STATE"; exit 1
fi
echo "COMPLETE output=$OUT log=$LOG" > "$STATE"
