#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
RUN="$TRAIN/logs/rsl_rl/g1/2026-09-25_03-56-03-Kick-a"
SNAP_DIR="$ROOT/artifacts/r3_interim_gpu1"
LOG="$ROOT/artifacts/monitoring/r3_interim_gpu1_model800.log"
STATE="$ROOT/artifacts/monitoring/r3_interim_gpu1_model800.state"
mkdir -p "$SNAP_DIR" "$ROOT/artifacts/monitoring"
used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi
cp "$RUN/model_800.pt" "$SNAP_DIR/model_800.pt"
if [[ ! -s "$SNAP_DIR/model_800.pt" ]]; then
  printf 'BLOCKED missing_snapshot\n' > "$STATE"
  exit 1
fi
printf 'RUNNING gpu=1 checkpoint=model_800.pt mode=interim num_envs=128 frames=300\n' > "$STATE"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TRAIN"
timeout --signal=TERM --kill-after=45s 1800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=128 --path="$SNAP_DIR/model_800.pt" \
  --headless --video --video_length=300 --id=r3_interim_gpu1 \
  > "$LOG" 2>&1
rc=$?
if (( rc != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
  printf 'BLOCKED exit=%s log=%s\n' "$rc" "$LOG" > "$STATE"
  exit 1
fi
if ! grep -q 'Success rate:' "$LOG"; then
  printf 'BLOCKED missing_success_metric log=%s\n' "$LOG" > "$STATE"
  exit 1
fi
metric="$(grep 'Success rate:' "$LOG" | tail -n 1)"
printf 'COMPLETE %s checkpoint=model_800.pt num_envs=128 frames=300\n' "$metric" > "$STATE"