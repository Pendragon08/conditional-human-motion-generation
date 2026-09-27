#!/usr/bin/env bash
set -euo pipefail

# Independent, inference-only preview. It never attaches to or pauses training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
SOURCE="$ROOT/third_party/CLoSD/output/CLoSD/r3_CLoSD_multitask_finetune_512_full_scene_v2/Humanoid.pth"
PREVIEW="$ROOT/third_party/CLoSD/output/CLoSD/r3_CLoSD_checkpoint_preview"
STATE="$MON/r3_closd_checkpoint_preview.state"
LOG="$MON/r3_closd_checkpoint_preview.log"
mkdir -p "$MON" "$PREVIEW"

[[ -s "$SOURCE" ]] || { echo "BLOCKED checkpoint_missing" > "$STATE"; exit 1; }
[[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:7000/static/)" == 200 ]] || {
  echo "BLOCKED scene_server_unavailable" > "$STATE"; exit 1;
}

# Read a stable copy so a concurrent training checkpoint write cannot corrupt the preview.
TMP="$PREVIEW/Humanoid.pth.partial.$$"
trap 'rm -f "$TMP"' EXIT
cp "$SOURCE" "$TMP"
EPOCH="$("$ROOT/.venv-pbhc-training/bin/python" - "$TMP" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint["epoch"]))
PY
)"
mv -f "$TMP" "$PREVIEW/Humanoid.pth"

export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/sim-web-visualizer:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1
export R3_CLOSD_WEB_FULL_SCENE=1 R3_CLOSD_WEB_MAX_ENVS=1 R3_CLOSD_WEB_RENDER_EVERY=16
cd "$ROOT/third_party/CLoSD"
echo "RUNNING inference_only checkpoint_epoch=$EPOCH gpu_physical=0 scene=meshcat training_unaffected=true" > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 1200s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid env=closd_multitask \
  exp_name=r3_CLoSD_checkpoint_preview epoch=-1 test=True \
  no_virtual_display=True headless=False env.num_envs=9 \
  env.episode_length=300 learning.params.config.player.games_num=900 \
  no_log=True debug=False \
  > "$LOG" 2>&1
RC=$?
set -e
if (( RC == 124 )); then
  echo "STOPPED time_limit inference_only checkpoint_epoch=$EPOCH" > "$STATE"
elif (( RC == 0 )); then
  echo "COMPLETE inference_only checkpoint_epoch=$EPOCH" > "$STATE"
else
  echo "FAILED exit=$RC inference_only checkpoint_epoch=$EPOCH log=$LOG" > "$STATE"
fi
exit "$RC"
