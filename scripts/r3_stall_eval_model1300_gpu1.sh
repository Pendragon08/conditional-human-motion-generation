#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
RUN="$TRAIN/logs/rsl_rl/g1/2026-09-25_03-56-03-Kick-a"
OUT="$ROOT/artifacts/r3_interim_curve_gpu1"
MON="$ROOT/artifacts/monitoring"
TRAIN_EXIT="$MON/r3_dreamcontrol_kick_formal_1024_2000.exit"
mkdir -p "$OUT" "$MON"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg_gpu1" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib_gpu1"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
for step in 1300; do
  STATE="$MON/r3_interim_curve_model${step}.state"
  LOG="$MON/r3_interim_curve_model${step}.log"
  CKPT="$RUN/model_${step}.pt"
  SNAP="$OUT/model_${step}/model_${step}.pt"
  if [[ -f "$STATE" ]] && grep -q '^COMPLETE ' "$STATE"; then
    continue
  fi
  printf 'WAITING_FOR_CHECKPOINT step=%s\n' "$step" > "$STATE"
  for _ in $(seq 1 600); do
    if [[ -s "$CKPT" ]]; then break; fi
    if [[ -f "$TRAIN_EXIT" ]] && [[ ! -s "$CKPT" ]]; then
      printf 'BLOCKED training_exited_before_checkpoint step=%s\n' "$step" > "$STATE"
      exit 1
    fi
    sleep 30
  done
  if [[ ! -s "$CKPT" ]]; then
    printf 'BLOCKED checkpoint_timeout step=%s\n' "$step" > "$STATE"
    exit 1
  fi
  s1="$(stat -c %s "$CKPT")"; sleep 5; s2="$(stat -c %s "$CKPT")"
  if [[ "$s1" != "$s2" ]]; then
    sleep 15
  fi
  if [[ ! -s "$CKPT" ]]; then
    printf 'BLOCKED checkpoint_unstable step=%s\n' "$step" > "$STATE"
    exit 1
  fi
  for _ in $(seq 1 120); do
    used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then break; fi
    sleep 30
  done
  if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
    printf 'BLOCKED gpu1_busy used_mib=%s step=%s\n' "$used" "$step" > "$STATE"
    exit 1
  fi
  mkdir -p "$(dirname "$SNAP")"
  cp "$CKPT" "$SNAP"
  printf 'RUNNING step=%s gpu=1 num_envs=128 frames=500\n' "$step" > "$STATE"
  cd "$TRAIN"
  timeout --signal=TERM --kill-after=45s 1800s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    scripts/reinforcement_learning/rsl_rl/play_eval.py \
    --task=Isaac-Motion-Tracking-Kick-v0 \
    --device=cuda:0 --num_envs=128 --path="$SNAP" \
    --headless --video --video_length=500 --id="r3_curve_${step}" \
    > "$LOG" 2>&1
  rc=$?
  if (( rc != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
    printf 'BLOCKED eval_exit=%s step=%s log=%s\n' "$rc" "$step" "$LOG" > "$STATE"
    exit 1
  fi
  if ! grep -q 'Success rate:' "$LOG"; then
    printf 'BLOCKED no_success_metric step=%s\n' "$step" > "$STATE"
    exit 1
  fi
  if ! find "$(dirname "$SNAP")/videos/play" -type f -name '*.mp4' -size +0c | grep -q .; then
    printf 'BLOCKED no_eval_video step=%s\n' "$step" > "$STATE"
    exit 1
  fi
  metric="$(grep 'Success rate:' "$LOG" | tail -1)"
  printf 'COMPLETE step=%s %s num_envs=128 frames=500\n' "$step" "$metric" > "$STATE"
done
printf 'COMPLETE steps=1300 num_envs=128 frames=500\n' > "$MON/r3_stall_eval_model1300_gpu1.state"