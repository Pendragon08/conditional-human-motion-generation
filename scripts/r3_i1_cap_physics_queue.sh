#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
CKPT="$OLD/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt"
STATE="$MON/r3_i1_cap_physics_queue.state"
echo 'WAITING_FOR_PAIRED_EVAL_AND_EXPORT' > "$STATE"
for _ in $(seq 1 240); do
  if grep -q '^COMPLETE caps=0.25,0.50' "$MON/r3_i1_cap_sweep_seed42.state" 2>/dev/null &&
     grep -q '^COMPLETE seeds=42,43,44 arms=raw,corrected' "$MON/r3_i1_physics_paired_queue.state" 2>/dev/null; then
    break
  fi
  if grep -q '^FAILED\|^BLOCKED' "$MON/r3_i1_cap_sweep_seed42.state" "$MON/r3_i1_physics_paired_queue.state" 2>/dev/null; then
    echo 'BLOCKED dependency_failed' > "$STATE"; exit 1
  fi
  sleep 30
done
grep -q '^COMPLETE caps=0.25,0.50' "$MON/r3_i1_cap_sweep_seed42.state" || {
  echo 'BLOCKED export_timeout' > "$STATE"; exit 1;
}
grep -q '^COMPLETE seeds=42,43,44 arms=raw,corrected' "$MON/r3_i1_physics_paired_queue.state" || {
  echo 'BLOCKED paired_eval_timeout' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
for TAG in 025 050; do
  REF="$ROOT/artifacts/i1_corrected_seed42_cap${TAG}"
  OUT="$ROOT/results/r3_i1_physics_seed42_cap${TAG}.json"
  LOG="$MON/r3_i1_physics_seed42_cap${TAG}.log"
  [[ ! -e "$OUT" && "$(find "$REF" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 64 ]] || {
    echo "BLOCKED output_or_ref_count tag=$TAG" > "$STATE"; exit 1;
  }
  for _ in $(seq 1 30); do
    USED="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1000 )) && break
    sleep 10
  done
  [[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1000 )) || {
    echo "BLOCKED gpu1_busy used_mib=$USED" > "$STATE"; exit 1;
  }
  export DREAMCONTROL_KICK_REF_PATH="$REF"
  echo "RUNNING cap=$TAG seed=42 envs=256 frames=500" > "$STATE"
  cd "$TRAIN"
  set +e
  timeout --signal=TERM --kill-after=45s 3600s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
    --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
    --video_length=500 --id="r3_i1_42_cap${TAG}" \
    --metrics-out="$OUT" > "$LOG" 2>&1
  RC=$?
  set -e
  if (( RC != 0 )) || [[ ! -s "$OUT" ]]; then
    echo "FAILED cap=$TAG exit=$RC log=$LOG" > "$STATE"; exit 1
  fi
done
echo "COMPLETE caps=0.25,0.50 seed=42 at=$(date -u +%FT%TZ)" > "$STATE"
