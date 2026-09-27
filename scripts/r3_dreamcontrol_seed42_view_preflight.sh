#!/usr/bin/env bash
set -euo pipefail

# Live GPU/stream view of the existing policy on the planned 98-reference set.
# No learning takes place. User confirmation is required before formal training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/augmented_train"
MON="$ROOT/artifacts/monitoring"
LOG="$MON/r3_dreamcontrol_seed42_view_preflight.log"
STATE="$MON/r3_dreamcontrol_seed42_view_preflight.state"
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 98 ]] || { echo 'BLOCKED reference_count' > "$STATE"; exit 1; }
export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$MON"
echo "STARTING refs=$REF envs=16 livestream=2 no_learning=true" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 1800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=16 \
  --path=logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a/model_1999.pt \
  --livestream=2 --id=r3_seed42_live_view \
  > "$LOG" 2>&1
rc=$?
set -e
echo "STOPPED exit=$rc log=$LOG" > "$STATE"
