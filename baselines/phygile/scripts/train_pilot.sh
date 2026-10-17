#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 20 existing reference clips; initial active expert only; not paper-scale GMT.
exec bash scripts/run_isaac.sh upstream/scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 --headless --device cuda:0 --num_envs 1024 \
  --seed 42 --max_iterations 500 --run_name train20_nostate_seed42 --logger tensorboard \
  --g1_training_cfg configs/moe_pilot.yaml --motion_dir data/train \
  --no_wandb_model_upload
