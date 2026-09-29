#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
CKPT=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt
STATE="$MON/r3_i1_live_training_view_probe.state"
LOG="$MON/r3_i1_live_training_view_probe.log"
mkdir -p "$MON"
exec 9>"$MON/r3_i1_live_training_view_probe.lock"
flock -n 9 || exit 1
[[ -s "$CKPT" ]] || { echo 'BLOCKED checkpoint_missing' > "$STATE"; exit 1; }
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo 'RUNNING inference_only_actual_simulation_pose_not_training' > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 1800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_i1_live_telemetry_policy_probe.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:1 \
  --num_envs=16 --frames=10000 --checkpoint="$CKPT" --headless \
  --telemetry-udp=127.0.0.1:40123 > "$LOG" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  echo 'COMPLETE inference_only_no_training' > "$STATE"
elif (( rc == 124 )); then
  echo 'COMPLETE expected_time_limit_inference_only_no_training' > "$STATE"
else
  echo "FAILED exit=$rc log=$LOG" > "$STATE"
  exit "$rc"
fi
