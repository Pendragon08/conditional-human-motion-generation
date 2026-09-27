#!/usr/bin/env bash
set -uo pipefail

# The original step-1600 queue timed out with the stalled run. This queue
# watches only the distinct resumed run and never touches GPU 0 training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
RUN="$TRAIN/logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a"
MON="$ROOT/artifacts/monitoring"
OUT="$ROOT/artifacts/r3_resume_interim_gpu1"
STATE="$MON/r3_resume_interim_1600_gpu1.state"
LOG="$MON/r3_resume_interim_1600_gpu1.log"
CKPT="$RUN/model_1600.pt"
SNAP="$OUT/model_1600/model_1600.pt"
TRAIN_EXIT="$MON/r3_dreamcontrol_kick_resume1300_700_1024.exit"
LOCK="$MON/r3_resume_interim_1600_gpu1.lock"

mkdir -p "$MON" "$OUT"
exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ -f "$STATE" ]] && grep -q '^COMPLETE ' "$STATE"; then exit 0; fi
printf 'WAITING_FOR_CHECKPOINT step=1600 run=%s\n' "$RUN" > "$STATE"
for _ in $(seq 1 720); do
  if [[ -s "$CKPT" ]]; then break; fi
  if [[ -f "$TRAIN_EXIT" ]]; then
    printf 'BLOCKED training_exited_before_checkpoint step=1600\n' > "$STATE"
    exit 1
  fi
  sleep 30
done
if [[ ! -s "$CKPT" ]]; then
  printf 'BLOCKED checkpoint_timeout step=1600\n' > "$STATE"
  exit 1
fi
size1="$(stat -c %s "$CKPT")"; sleep 5; size2="$(stat -c %s "$CKPT")"
if [[ "$size1" != "$size2" || ! -s "$CKPT" ]]; then
  printf 'BLOCKED checkpoint_unstable step=1600\n' > "$STATE"
  exit 1
fi
used=''
for _ in $(seq 1 120); do
  used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then break; fi
  sleep 30
done
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi

mkdir -p "$(dirname "$SNAP")"
cp "$CKPT" "$SNAP"
printf 'RUNNING step=1600 gpu=1 num_envs=128 frames=500\n' > "$STATE"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg_gpu1" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib_gpu1"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TRAIN"
timeout --signal=TERM --kill-after=45s 1800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=128 --path="$SNAP" \
  --headless --video --video_length=500 --id=r3_resume_curve_1600 \
  > "$LOG" 2>&1
result=$?
if (( result != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
  printf 'BLOCKED eval_exit=%s step=1600 log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
if ! grep -q 'Success rate:' "$LOG"; then
  printf 'BLOCKED no_success_metric step=1600\n' > "$STATE"
  exit 1
fi
if ! find "$(dirname "$SNAP")/videos/play" -type f -name '*.mp4' -size +0c | grep -q .; then
  printf 'BLOCKED no_eval_video step=1600\n' > "$STATE"
  exit 1
fi
metric="$(grep 'Success rate:' "$LOG" | tail -n 1)"
printf 'COMPLETE step=1600 %s num_envs=128 frames=500\n' "$metric" > "$STATE"
