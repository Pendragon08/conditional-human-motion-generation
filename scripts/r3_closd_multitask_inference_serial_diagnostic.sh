#!/usr/bin/env bash
set -uo pipefail

# Bounded diagnostic: serializes CLoSD motion loading with debug=True after
# the first official 9-env inference stalled at multiprocessing queue.get().
# This is inference only; it must not be reported as an RL-training Gate pass.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_multitask_serial_diagnostic.state"
LOG="$MON/r3_closd_multitask_serial_diagnostic.log"
ASSET="$ROOT/artifacts/closd-r3/official_assets_manifest.json"
LOCK="$MON/r3_closd_multitask_serial_diagnostic.lock"
mkdir -p "$MON" "$ROOT/cache/closd-r3-torch-extensions"
exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ ! -s "$ASSET" ]]; then
  printf 'BLOCKED missing_official_asset_manifest\n' > "$STATE"
  exit 1
fi
used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
cd "$ROOT/third_party/CLoSD"
printf 'RUNNING started=%s gpu=1 envs=9 debug=True serial_motion_load=true\n' "$(date -u +%FT%TZ)" > "$STATE"
timeout --signal=TERM --kill-after=30s 1500s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid \
  epoch=-1 test=True no_virtual_display=True \
  headless=True env.num_envs=9 env.episode_length=300 \
  env=closd_multitask exp_name=CLoSD_multitask_finetune \
  learning.params.config.player.games_num=9 \
  debug=True \
  > "$LOG" 2>&1
result=$?
if (( result != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
  printf 'BLOCKED inference_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'FINISHED_NEEDS_VALIDATION exit=0 log=%s\n' "$LOG" > "$STATE"
