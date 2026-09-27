#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_capacity_512_inference.state"
LOG="$MON/r3_closd_capacity_512_inference.log"
[[ -s "$ROOT/artifacts/closd-r3/official_assets_manifest.json" ]] || {
  echo 'BLOCKED assets_missing' > "$STATE"; exit 1;
}
while tmux has-session -t r3_closd_visual_preflight 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1
export R3_CLOSD_TELEMETRY_UDP=127.0.0.1:18445
cd "$ROOT/third_party/CLoSD"
echo 'RUNNING inference_capacity_only envs=512 episode_length=20 no_training=true' > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 900s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid env=closd_multitask \
  exp_name=CLoSD_multitask_finetune epoch=-1 test=True \
  no_virtual_display=True headless=True env.num_envs=512 \
  env.episode_length=20 learning.params.config.player.games_num=1 \
  no_log=True debug=False > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  echo "BLOCKED inference_capacity_exit=$rc log=$LOG" > "$STATE"; exit 1
fi
echo "COMPLETE inference_capacity_only envs=512 log=$LOG" > "$STATE"
