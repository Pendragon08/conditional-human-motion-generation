#!/usr/bin/env bash
set -uo pipefail

# Evaluates only the separate resumed run after its successful completion.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
PREFIX="$MON/r3_dreamcontrol_kick_resume1300_700_1024"
TRAIN_LOG="$PREFIX.log"
TRAIN_EXIT="$PREFIX.exit"
STATE="$MON/r3_resume_posttrain_eval.state"
EVAL_LOG="$MON/r3_resume_posttrain_eval.log"
LOCK="$MON/r3_resume_posttrain_eval.lock"

exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ -f "$STATE" ]] && grep -q '^COMPLETE ' "$STATE"; then exit 0; fi
printf 'WAITING_FOR_TRAINING %s\n' "$(date -u +%FT%TZ)" > "$STATE"
while [[ ! -f "$TRAIN_EXIT" ]]; do sleep 60; done

status="$(tr -d '[:space:]' < "$TRAIN_EXIT")"
if [[ "$status" != 0 ]]; then
  printf 'BLOCKED training_exit=%s\n' "$status" > "$STATE"
  exit 1
fi
if ! grep -q 'Learning iteration 1999/2000' "$TRAIN_LOG"; then
  printf 'BLOCKED missing_final_iteration\n' > "$STATE"
  exit 1
fi
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$TRAIN_LOG" | tail -n 1 | tr -d '\r')"
run="$TRAIN/logs/rsl_rl/g1/$run_name"
if [[ -z "$run_name" || ! -d "$run" ]]; then
  printf 'BLOCKED missing_resume_run log=%s\n' "$TRAIN_LOG" > "$STATE"
  exit 1
fi
checkpoint="$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
  printf 'BLOCKED no_checkpoint run=%s\n' "$run" > "$STATE"
  exit 1
fi

printf 'WAITING_FOR_GPU checkpoint=%s\n' "$checkpoint" > "$STATE"
used=''
for _ in $(seq 1 120); do
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then break; fi
  sleep 30
done
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu0_not_free used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi

printf 'RUNNING checkpoint=%s\n' "$checkpoint" > "$STATE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TRAIN"
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=1000 --path="$checkpoint" \
  --livestream=2 --video --video_length=500 --id=r3_resume_final \
  > "$EVAL_LOG" 2>&1
result=$?
if (( result != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$EVAL_LOG"; then
  printf 'BLOCKED eval_exit=%s log=%s\n' "$result" "$EVAL_LOG" > "$STATE"
  exit 1
fi
if ! grep -q 'Success rate:' "$EVAL_LOG"; then
  printf 'BLOCKED no_success_metric log=%s\n' "$EVAL_LOG" > "$STATE"
  exit 1
fi
if ! find "$run/videos/play" -type f -name '*.mp4' -size +0c | grep -q .; then
  printf 'BLOCKED no_eval_video run=%s\n' "$run" > "$STATE"
  exit 1
fi
metric="$(grep 'Success rate:' "$EVAL_LOG" | tail -n 1)"
printf 'COMPLETE %s checkpoint=%s\n' "$metric" "$checkpoint" > "$STATE"
