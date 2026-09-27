#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_full_scene_preflight.state"
LOG="$MON/r3_closd_full_scene_preflight.log"
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/sim-web-visualizer:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1
export R3_CLOSD_WEB_FULL_SCENE=1 R3_CLOSD_WEB_MAX_ENVS=2
cd "$ROOT/third_party/CLoSD"
printf 'RUNNING inference_only full_isaac_scene=true envs=9 gpu=1\n' > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 900s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid env=closd_multitask \
  exp_name=CLoSD_multitask_finetune epoch=-1 test=True \
  no_virtual_display=True headless=False env.num_envs=9 \
  env.episode_length=300 learning.params.config.player.games_num=900 \
  no_log=True debug=False \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc == 124 )); then
  printf 'STOPPED time_limit inference_only; training_not_started\n' > "$STATE"
elif (( rc == 0 )); then
  printf 'COMPLETE inference_only; training_not_started\n' > "$STATE"
else
  printf 'FAILED exit=%s log=%s\n' "$rc" "$LOG" > "$STATE"
fi
exit "$rc"
