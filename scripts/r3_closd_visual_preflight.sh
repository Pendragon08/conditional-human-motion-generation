#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_visual_preflight.state"
LOG="$MON/r3_closd_visual_preflight.log"
ASSETS="$ROOT/artifacts/closd-r3/official_assets_manifest.json"
[[ -s "$ASSETS" ]] || { echo 'BLOCKED assets_missing' > "$STATE"; exit 1; }
[[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6011/state)" == 200 ]] || {
  echo 'BLOCKED local_dashboard_unavailable' > "$STATE"; exit 1;
}
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
echo 'RUNNING inference_only envs=9 gpu=1 live_telemetry=true no_training=true' > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 1800s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid env=closd_multitask \
  exp_name=CLoSD_multitask_finetune epoch=-1 test=True \
  no_virtual_display=True headless=True env.num_envs=9 \
  env.episode_length=300 learning.params.config.player.games_num=900 \
  no_log=True debug=False \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  echo "BLOCKED inference_exit=$rc log=$LOG" > "$STATE"; exit 1
fi
echo "COMPLETE inference_only log=$LOG" > "$STATE"
