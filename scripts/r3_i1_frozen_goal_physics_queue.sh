#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_frozen_goal_physics_queue.state"
RAW="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42/Kick_sim1"
CKPT="$OLD/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt"
exec 9>"$MON/r3_i1_frozen_goal_physics_queue.lock"
flock -n 9 || { echo 'already running' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
[[ -s "$CKPT" && -d "$RAW" ]] || { echo 'BLOCKED missing_input' > "$STATE"; exit 1; }
for _ in $(seq 1 60); do
  USED="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1000 )) && break
  sleep 10
done
[[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1000 )) || {
  echo "BLOCKED gpu1_busy used_mib=$USED" > "$STATE"; exit 1;
}
SMOKE="$ROOT/results/r3_i1_frozen_goal_smoke.json"
[[ ! -e "$SMOKE" ]] || { echo "BLOCKED smoke_output_exists" > "$STATE"; exit 1; }
export DREAMCONTROL_KICK_REF_PATH="$ROOT/artifacts/i1_corrected_seed42_cap025"
echo 'RUNNING smoke corrected_cap025 freeze_goal=raw envs=8 frames=50' > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 1200s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
  --num_envs=8 --eval-seed=42 --path="$CKPT" --headless --video \
  --video_length=50 --id=r3_i1_frozen_smoke \
  --freeze-task-goal-from-dir="$RAW" --metrics-out="$SMOKE" \
  > "$MON/r3_i1_frozen_goal_smoke.log" 2>&1
RC=$?
set -e
if (( RC != 0 )) || [[ ! -s "$SMOKE" ]]; then
  echo "FAILED smoke_exit=$RC output_present=$([[ -s "$SMOKE" ]] && echo yes || echo no)" > "$STATE"; exit 1
fi
for ARM in raw cap025 cap050 cap100; do
  case "$ARM" in
    raw) REF="$RAW";;
    cap025) REF="$ROOT/artifacts/i1_corrected_seed42_cap025";;
    cap050) REF="$ROOT/artifacts/i1_corrected_seed42_cap050";;
    cap100) REF="$ROOT/artifacts/i1_corrected_seed42";;
  esac
  OUT="$ROOT/results/r3_i1_frozen_goal_seed42_${ARM}.json"
  LOG="$MON/r3_i1_frozen_goal_seed42_${ARM}.log"
  [[ ! -e "$OUT" ]] || { echo "BLOCKED output_exists=$ARM" > "$STATE"; exit 1; }
  export DREAMCONTROL_KICK_REF_PATH="$REF"
  echo "RUNNING arm=$ARM seed=42 envs=256 frames=500 goal_from=raw" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=45s 3600s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
    --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
    --video_length=500 --id="r3_i1_frozen_${ARM}" \
    --freeze-task-goal-from-dir="$RAW" --metrics-out="$OUT" > "$LOG" 2>&1
  RC=$?
  set -e
  if (( RC != 0 )) || [[ ! -s "$OUT" ]]; then
    echo "FAILED arm=$ARM exit=$RC log=$LOG" > "$STATE"; exit 1
  fi
done
echo "COMPLETE arms=raw,cap025,cap050,cap100 seed=42 at=$(date -u +%FT%TZ)" > "$STATE"
