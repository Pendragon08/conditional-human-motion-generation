#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
TRAIN_STATE="$MON/r3_dreamcontrol_official_kick_8192_2000.state"
TRAIN_LOG="$MON/r3_dreamcontrol_official_kick_8192_2000.log"
LOG="$MON/r3_dreamcontrol_official_kick_8192_eval.log"
STATE="$MON/r3_dreamcontrol_official_kick_8192_eval.state"
grep -q '^COMPLETE iteration=1999/2000' "$TRAIN_STATE" || {
  echo 'BLOCKED official_training_not_complete' > "$STATE"; exit 1;
}
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$TRAIN_LOG" | tail -n 1 | tr -d '\r')"
checkpoint="$TRAIN/logs/rsl_rl/g1/$run_name/model_1999.pt"
[[ -n "$run_name" && -s "$checkpoint" ]] || {
  echo 'BLOCKED final_checkpoint_missing' > "$STATE"; exit 1;
}
unset DREAMCONTROL_KICK_REF_PATH CUDA_VISIBLE_DEVICES PUBLIC_IP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo "RUNNING checkpoint=$checkpoint refs=official66 envs=1000 frames=500" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=1000 --path="$checkpoint" \
  --headless --video --video_length=500 --id=r3_official_kick_8192_eval \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Success rate:' "$LOG"; then
  echo "BLOCKED eval_exit=$rc log=$LOG" > "$STATE"; exit 1
fi
video="$(find "$TRAIN/logs/rsl_rl/g1/$run_name/videos/play" -type f -name '*r3_official_kick_8192_eval*.mp4' -size +0c | head -n 1)"
[[ -n "$video" ]] || { echo "BLOCKED no_video log=$LOG" > "$STATE"; exit 1; }
metric="$(grep 'Success rate:' "$LOG" | tail -n 1)"
echo "COMPLETE $metric checkpoint=$checkpoint video=$video" > "$STATE"
