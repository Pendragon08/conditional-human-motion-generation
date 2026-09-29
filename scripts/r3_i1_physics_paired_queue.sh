#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
CKPT="$OLD/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt"
STATE="$MON/r3_i1_physics_paired_queue.state"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_physics_paired_queue.lock"
flock -n 9 || { echo 'already running' >&2; exit 1; }
[[ -s "$CKPT" ]] || { echo "BLOCKED missing_checkpoint" > "$STATE"; exit 1; }
grep -q '^COMPLETE seeds=42,43,44 outputs=191' "$MON/r3_i1_export_corrected_refs.state" || {
  echo 'BLOCKED corrected_references_not_complete' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
for SEED in 42 43 44; do
  for ARM in raw corrected; do
    if [[ "$ARM" == raw ]]; then
      REF="$ROOT/third_party/DreamControl/TrajGen/sample_i1_fixed_seed${SEED}/Kick_sim1"
    else
      REF="$ROOT/artifacts/i1_corrected_seed${SEED}"
    fi
    case "$SEED" in 42|43) EXPECTED=64;; 44) EXPECTED=63;; esac
    COUNT="$(find "$REF" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
    [[ "$COUNT" -eq "$EXPECTED" ]] || {
      echo "BLOCKED seed=$SEED arm=$ARM ref_count=$COUNT expected=$EXPECTED" > "$STATE"
      exit 1
    }
    OUT="$ROOT/results/r3_i1_physics_seed${SEED}_${ARM}.json"
    LOG="$MON/r3_i1_physics_seed${SEED}_${ARM}.log"
    [[ ! -e "$OUT" ]] || { echo "BLOCKED output_exists=$OUT" > "$STATE"; exit 1; }
    export DREAMCONTROL_KICK_REF_PATH="$REF"
    echo "RUNNING seed=$SEED arm=$ARM envs=256 frames=500 fixed_checkpoint=$CKPT" > "$STATE"
    cd "$TRAIN"
    set +e
    timeout --signal=TERM --kill-after=45s 3600s \
      "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
      "$ROOT/scripts/r3_i1_play_eval_metrics.py" \
      --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
      --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
      --video_length=500 --id="r3_i1_${SEED}_${ARM}" \
      --metrics-out="$OUT" > "$LOG" 2>&1
    RC=$?
    set -e
    if (( RC != 0 )) || [[ ! -s "$OUT" ]]; then
      echo "FAILED seed=$SEED arm=$ARM exit=$RC log=$LOG" > "$STATE"
      exit 1
    fi
  done
done
python3 "$ROOT/scripts/r3_i1_physics_paired_summary.py" \
  --results "$ROOT/results" --output "$ROOT/results/r3_i1_physics_paired_summary.json" \
  > "$MON/r3_i1_physics_paired_summary.log" 2>&1 || {
    echo "FAILED paired_summary" > "$STATE"; exit 1;
  }
echo "COMPLETE seeds=42,43,44 arms=raw,corrected at=$(date -u +%FT%TZ)" > "$STATE"
