#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
REF="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed42/Kick_sim1"
CKPT="$OLD/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt"
STATE="$MON/r3_i1_physics_smoke.state"
LOG="$MON/r3_i1_physics_smoke.log"
OUT="$ROOT/results/r3_i1_physics_smoke.json"
mkdir -p "$MON" "$ROOT/results"
[[ -s "$CKPT" ]] || { echo "BLOCKED missing_checkpoint" > "$STATE"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "BLOCKED metrics_exist=$OUT" > "$STATE"; exit 1; }
export CUDA_VISIBLE_DEVICES=1 DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo "RUNNING refs=raw_seed42 envs=8 frames=50 gpu=1" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 1200s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
  --num_envs=8 --eval-seed=42 --path="$CKPT" --headless --video \
  --video_length=50 --id=r3_i1_physics_smoke \
  --metrics-out="$OUT" \
  > "$LOG" 2>&1
RC=$?
set -e
if (( RC != 0 )) || [[ ! -s "$OUT" ]]; then
  echo "FAILED exit=$RC log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE exit=0 metrics=$OUT" > "$STATE"
