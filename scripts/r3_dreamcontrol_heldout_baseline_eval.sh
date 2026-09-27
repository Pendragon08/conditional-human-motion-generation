#!/usr/bin/env bash
set -euo pipefail

# Fixed held-out diagnostic for the completed base66 policy. No training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
RUN="$TRAIN/logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/generated_holdout"
LOG="$MON/r3_dreamcontrol_heldout_baseline_eval.log"
STATE="$MON/r3_dreamcontrol_heldout_baseline_eval.state"
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 32 ]] || { echo 'BLOCKED holdout_count' > "$STATE"; exit 1; }
[[ -s "$RUN/model_1999.pt" ]] || { echo 'BLOCKED checkpoint' > "$STATE"; exit 1; }

export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$MON"
echo "RUNNING base66_policy=$RUN/model_1999.pt heldout32=$REF envs=1000 frames=500" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=1000 --path="$RUN/model_1999.pt" \
  --livestream=2 --video --video_length=500 --id=r3_seed42_holdout_base66 \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Success rate:' "$LOG"; then
  echo "BLOCKED eval_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
metric="$(grep 'Success rate:' "$LOG" | tail -n 1)"
echo "COMPLETE $metric checkpoint=$RUN/model_1999.pt refs=$REF" > "$STATE"
