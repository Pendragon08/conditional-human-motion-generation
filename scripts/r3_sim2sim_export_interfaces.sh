#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
OUT="$ROOT/artifacts/sim2sim_mujoco/interfaces"
mkdir -p "$MON" "$OUT" "$ROOT/cache/r3_tmp"
exec 9>"$MON/r3_sim2sim_interface_export.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$ROOT/cache/r3_tmp" WANDB_MODE=disabled
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl"
cd "$TRAIN"
for task in Kick Punch; do
  if [[ "$task" == Kick ]]; then
    policy="/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/exported/policy.pt"
  else
    policy="$TRAIN/logs/rsl_rl/g1/2026-09-29_12-37-46-Punch-a/exported/policy.pt"
  fi
  echo "RUNNING task=$task read_only=true" > "$MON/r3_sim2sim_interface_export.state"
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u "$ROOT/scripts/r3_sim2sim_isaac_export.py" \
    --task "$task" --policy "$policy" --output "$OUT" --device cuda:0 --headless \
    > "$MON/r3_sim2sim_${task}_interface_export.log" 2>&1
done
echo 'COMPLETE tasks=Kick,Punch snapshots=4_each learning=false' > "$MON/r3_sim2sim_interface_export.state"
