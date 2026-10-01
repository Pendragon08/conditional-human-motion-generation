#!/usr/bin/env bash
# Inference-only live-scene acceptance; no learning, no new evaluation score.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd:$RUNTIME/third_party/sim-web-visualizer"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
export R3_CLOSD_SCENE_UDP=127.0.0.1:18446
exec 9>"$MON/r3_closd_reproduction_20261001.lock"
flock -n 9 || { echo 'Formal queue is running; do not add a second producer'; exit 1; }
STATE="$MON/r3_closd_visibility_preflight_20261001.state"
LOG="$MON/r3_closd_visibility_preflight_20261001.log"
[[ ! -e "$LOG" ]] || { echo 'Existing preflight requires audit before restart'; exit 1; }
cd "$ROOT/third_party/CLoSD"
echo 'RUNNING inference_only envs=9 no_learning=true' > "$STATE"
set +e
timeout --signal=TERM --kill-after=45s 1800s "$RUNTIME/.venv-pbhc-training/bin/python" -u \
  "$ROOT/scripts/r3_closd_isolated_entry.py" learning=im_big env=closd_multitask robot=smpl_humanoid \
  exp_name=CLoSD_multitask_finetune epoch=-1 test=True env.episode_length=500 env.dip.cfg_param=7.5 \
  env.num_envs=9 no_virtual_display=True headless=True closd_eval=False \
  learning.params.config.player.games_num=100000 no_log=True debug=False seed=0 > "$LOG" 2>&1
rc=$?
set -e
if [[ "$rc" == 124 ]]; then
  echo 'STOPPED deadline_exit=124 inference_only=true no_training_interrupted=true' > "$STATE"
elif [[ "$rc" == 0 ]]; then
  echo 'COMPLETE inference_only=true no_formal_score=true' > "$STATE"
else
  echo "FAILED inference_exit=$rc no_training_interrupted=true" > "$STATE"
  exit "$rc"
fi
