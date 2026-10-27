#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
updates="${1:-1500}"
num_envs="${2:-1024}"
run_name="${3:-extended2000_seed42}"
export PHYGILE_RESUME_PARENT_SHA256=afad21ce7192dc4ec10e04fd2bc7684e9278f901dbffcb053cf45b8b04bda162
exec bash scripts/run_isaac.sh scripts/resume_training.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 --headless --device cuda:0 --num_envs "$num_envs" \
  --seed 42 --max_iterations "$updates" --run_name "$run_name" --logger tensorboard \
  --g1_training_cfg configs/moe_pilot.yaml --motion_dir data/train \
  --experiment_name phygile_moe_pilot --resume True \
  --load_run 2026-10-17_09-16-39_train20_nostate_seed42 --checkpoint model_499.pt \
  --no_wandb_model_upload
