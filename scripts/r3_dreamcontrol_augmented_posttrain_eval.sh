#!/usr/bin/env bash
set -euo pipefail

# Fixed held-out evaluation after the seed42-augmented 2000-iteration run.
# Render the saved video offscreen; WebRTC is handled by a separate viewer.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
TRAIN_STATE="${R3_AUG_TRAIN_STATE:-$MON/r3_dreamcontrol_augmented_seed42_2000_1024.state}"
TRAIN_LOG="${R3_AUG_TRAIN_LOG:-$MON/r3_dreamcontrol_augmented_seed42_2000_1024.log}"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/generated_holdout"
LOG="$MON/r3_dreamcontrol_augmented_seed42_heldout_eval.log"
STATE="$MON/r3_dreamcontrol_augmented_seed42_heldout_eval.state"

if ! grep -q '^COMPLETE iteration=1999/2000' "$TRAIN_STATE"; then
  echo "BLOCKED training_not_complete" > "$STATE"
  exit 1
fi
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 32 ]] || { echo 'BLOCKED holdout_count' > "$STATE"; exit 1; }
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$TRAIN_LOG" | tail -n 1 | tr -d '\r')"
RUN="$TRAIN/logs/rsl_rl/g1/$run_name"
[[ -n "$run_name" && -d "$RUN" ]] || { echo 'BLOCKED run_directory' > "$STATE"; exit 1; }
CHECKPOINT="$RUN/model_1999.pt"
[[ -s "$CHECKPOINT" ]] || { echo 'BLOCKED final_checkpoint' > "$STATE"; exit 1; }

echo "WAITING_FOR_GPU checkpoint=$CHECKPOINT" > "$STATE"
for _ in $(seq 1 120); do
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )) && break
  sleep 30
done
[[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )) || { echo "BLOCKED gpu0_busy used_mib=$used" > "$STATE"; exit 1; }

export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo "RUNNING checkpoint=$CHECKPOINT refs=$REF envs=1000 frames=500" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=1000 --path="$CHECKPOINT" \
  --headless --video --video_length=500 --id=r3_aug_seed42_holdout \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Success rate:' "$LOG"; then
  echo "BLOCKED eval_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
if ! find "$RUN/videos/play" -name '*r3_aug_seed42_holdout*.mp4' -size +0c | grep -q .; then
  echo "BLOCKED no_video log=$LOG" > "$STATE"
  exit 1
fi
metric="$(grep 'Success rate:' "$LOG" | tail -n 1)"
echo "COMPLETE $metric checkpoint=$CHECKPOINT refs=$REF" > "$STATE"
