#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_isaac.sh upstream/scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 --headless --device cuda:0 --num_envs 32 \
  --seed 42 --max_iterations 1 --run_name smoke --logger tensorboard \
  --g1_training_cfg configs/moe_pilot.yaml --motion_file data/train/sample_000_actor0.npz \
  --no_wandb_model_upload
