#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_semantic_contact_queue.state"
mkdir -p "$MON"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget${PYTHONPATH:+:$PYTHONPATH}"
for SEED in 42 43 44; do
  echo "RUNNING seed=$SEED" > "$STATE"
  if ! timeout 600s "$OLD/.venv-retarget/bin/python" -u \
    "$ROOT/scripts/r3_i1_semantic_contact_proxy.py" \
    --stage "$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed${SEED}" \
    --corrected-dir "$ROOT/artifacts/i1_corrected_seed${SEED}" \
    --output "$ROOT/results/r3_i1_semantic_contact_proxy_seed${SEED}.json" \
    > "$MON/r3_i1_semantic_contact_seed${SEED}.log" 2>&1; then
    echo "FAILED seed=$SEED" > "$STATE"
    exit 1
  fi
done
echo "COMPLETE seeds=42,43,44 at=$(date -u +%FT%TZ)" > "$STATE"
