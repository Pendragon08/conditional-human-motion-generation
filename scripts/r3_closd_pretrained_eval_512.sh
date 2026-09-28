#!/usr/bin/env bash
set -euo pipefail

# Matched evaluation for the official no-finetune controller, not a training job.
ROOT="${R3_RUNTIME_ROOT:-/data/code/ouyanghao/oyh/humanoid-combat-motion}"
MON="$ROOT/artifacts/monitoring"
SRC="$ROOT/third_party/CLoSD/output/CLoSD/CLoSD_no_finetune/Humanoid.pth"
DEST="$ROOT/third_party/CLoSD/output/CLoSD/r3_CLoSD_pretrained_eval_512"
STATE="$MON/r3_closd_pretrained_eval_512.state"
LOG="$MON/r3_closd_pretrained_eval_512.log"

mkdir -p "$MON" "$DEST"
[[ -s "$SRC" ]] || { echo 'BLOCKED pretrained_checkpoint_missing' > "$STATE"; exit 1; }
[[ -s "$ROOT/artifacts/closd-r3/official_assets_manifest.json" ]] || {
  echo 'BLOCKED official_assets_missing' > "$STATE"; exit 1;
}
if [[ -e "$DEST/Humanoid.pth" ]]; then
  cmp -s "$SRC" "$DEST/Humanoid.pth" || {
    echo 'BLOCKED eval_checkpoint_conflict' > "$STATE"; exit 1;
  }
else
  cp -L "$SRC" "$DEST/Humanoid.pth"
fi
CKPT_SHA="$(sha256sum "$SRC" | cut -d ' ' -f 1)"

export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1

cd "$ROOT/third_party/CLoSD"
printf 'RUNNING pretrained_eval envs=512 episode_length=500 cfg=7.5 checkpoint_sha256=%s\n' "$CKPT_SHA" > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 43200s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big env=closd_multitask robot=smpl_humanoid \
  exp_name=r3_CLoSD_pretrained_eval_512 epoch=-1 test=True \
  env.episode_length=500 env.dip.cfg_param=7.5 env.num_envs=512 \
  no_virtual_display=True headless=True closd_eval=True \
  learning.params.config.player.games_num=100000 \
  no_log=True debug=False seed=0 > "$LOG" 2>&1
RC=$?
set -e
if (( RC == 0 )) && grep -q 'Eval plot ended' "$LOG"; then
  printf 'COMPLETE pretrained_eval envs=512 checkpoint_sha256=%s log=%s\n' "$CKPT_SHA" "$LOG" > "$STATE"
else
  printf 'FAILED pretrained_eval exit=%s checkpoint_sha256=%s log=%s\n' "$RC" "$CKPT_SHA" "$LOG" > "$STATE"
  exit 1
fi
