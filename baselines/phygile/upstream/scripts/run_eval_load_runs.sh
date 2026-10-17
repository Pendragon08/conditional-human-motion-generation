#!/usr/bin/env bash
# bash /ssd/bjc/bm_mlp/scripts/run_eval_load_runs.sh

set -euo pipefail

cd /ssd/bjc/bm_mlp

runs=(
  # '2025-11-25_17-27-57_accad_bm0_future'
  # '2025-11-25_17-28-23_accad_bm_encoder'
  # '2025-11-25_17-40-40_accad_bm2_encoder_future'
  # '2025-11-25_17-47-50_accad_bm3_future_advnorm'
  # '2025-11-25_17-50-42_accad_bm4_encoder_advnorm'
  # '2025-11-25_17-53-42_accad_bm5_encoder_future_advnorm'
  '2025-11-30_00-15-42_accad_bm0_future_success'
  '2025-11-30_00-21-37_accad_bm_encoder_success'
  '2025-11-30_00-24-53_accad_bm2_encoder_future_success'
  '2025-11-30_00-31-18_accad_bm3_future_adv_success'
  '2025-11-30_00-33-50_accad_bm4_encoder_future_adv_success'
#   '2025-12-01_13-51-46_accad_bm5_future_static0.2_nosuccess'
)

for run in "${runs[@]}"; do
  printf '\n[INFO] running eval for %s\n' "$run"
  python scripts/rsl_rl/eval_metrics_folder.py \
    --task Tracking-Flat-G1-v0 \
    --num_envs 10000 \
    --device cuda:0 \
    --load_run "$run" \
    --checkpoint model_32000.pt \
    --mpjpe_print_threshold 0.28 \
    --motion_folder /ssd/humanoid/dataset/test_multi_accad \
    --headless \
    --eval_runs 50
done

