#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
REF="$ROOT/artifacts/r3_i1_generation_joint_v2_paired_refs"
GOALS="$ROOT/results/r3_i1_generation_joint_v2_paired_goals.json"
CKPT=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt
STATE="$MON/r3_i1_generation_joint_v2_fixed_tracker_eval.state"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_generation_joint_v2_fixed_tracker_eval.lock"
flock -n 9 || { echo 'Evaluation already active' >&2; exit 1; }
[[ -s "$CKPT" && -s "$GOALS" ]] || { echo 'BLOCKED checkpoint_or_goals_missing' > "$STATE"; exit 1; }
expected="$(find -L "$REF/baseline" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
[[ "$expected" -gt 0 && "$expected" -eq "$(find -L "$REF/candidate" -maxdepth 1 -name '*.pkl' -type f | wc -l)" ]] || {
  echo 'BLOCKED unmatched_reference_counts' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
for arm in baseline candidate; do
  export DREAMCONTROL_KICK_REF_PATH="$REF/$arm"
  OUT="$ROOT/results/r3_i1_generation_joint_v2_${arm}_fixed_tracker_eval.json"
  LOG="$MON/r3_i1_generation_joint_v2_${arm}_fixed_tracker_eval.log"
  [[ ! -e "$OUT" ]] || { echo "BLOCKED output_exists=$OUT" > "$STATE"; exit 1; }
  echo "RUNNING arm=$arm envs=256 frames=500 seed=42 external_fixed_goal=true tracker=fixed" > "$STATE"
  cd "$TRAIN"
  set +e
  timeout --signal=TERM --kill-after=45s 3600s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_i1_play_eval_external_goal.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:0 \
    --num_envs=256 --eval-seed=42 --path="$CKPT" --headless --video \
    --video_length=500 --id="r3_i1_generation_joint_v2_${arm}" \
    --external-goals-json="$GOALS" --metrics-out="$OUT" \
    > "$LOG" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )) || [[ ! -s "$OUT" ]]; then
    echo "FAILED arm=$arm exit=$rc log=$LOG" > "$STATE"; exit 1
  fi
done
echo "COMPLETE arms=baseline,candidate paired_refs=$expected external_fixed_goal=true" > "$STATE"
