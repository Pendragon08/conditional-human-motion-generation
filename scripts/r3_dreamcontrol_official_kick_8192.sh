#!/usr/bin/env bash
set -euo pipefail

# Faithful numerical baseline for DreamControl Precise Kick. The paper states
# 8192 parallel environments and 2000 PPO iterations on an RTX A6000.
# This uses the official 66 Kick_sim2 references and official train.py.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
REF="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
LOG="$MON/r3_dreamcontrol_official_kick_8192_2000.log"
STATE="$MON/r3_dreamcontrol_official_kick_8192_2000.state"
mkdir -p "$MON"

[[ "$(find "$REF" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 66 ]] || {
  echo 'BLOCKED official_reference_count_not_66' > "$STATE"; exit 1;
}
[[ -s "$TRAIN/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd" ]] || {
  echo 'BLOCKED robot_asset_missing' > "$STATE"; exit 1;
}

# The only local edit to Kick config is an optional reference-path override;
# leaving it unset preserves the upstream ../TrajGen/sample/Kick_sim2 default.
unset DREAMCONTROL_KICK_REF_PATH CUDA_VISIBLE_DEVICES PUBLIC_IP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PATH="$ROOT/cache/git-lfs-3.8.0:$PATH"
[[ -x "$ROOT/cache/git-lfs-3.8.0/git-lfs" ]] || {
  echo 'BLOCKED verified_git_lfs_missing' > "$STATE"; exit 1;
}
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

echo "RUNNING task=Kick refs=official66 envs=8192 iterations=2000 seed=42 trainer=upstream_train.py gpu=0 headless=true" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 172800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --num_envs=8192 --max_iterations=2000 --seed=42 --headless \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  echo "BLOCKED train_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$LOG" | tail -n 1 | tr -d '\r')"
checkpoint="$TRAIN/logs/rsl_rl/g1/$run_name/model_1999.pt"
[[ -n "$run_name" && -s "$checkpoint" ]] || {
  echo "BLOCKED final_checkpoint_missing run=$run_name" > "$STATE"; exit 1;
}
echo "COMPLETE iteration=1999/2000 checkpoint=$checkpoint log=$LOG" > "$STATE"
