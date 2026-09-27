#!/usr/bin/env bash
set -euo pipefail

# Short, non-baseline GPU GUI test for the next DreamControl task.  This runs
# the actual official training environment with a small resource budget; it
# never counts as the paper's 8192-environment/2000-iteration reproduction.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
REF="$ROOT/third_party/DreamControl/TrajGen/sample/Punch_sim2"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_punch_gui_preflight.state"
LOG="$MON/r3_dreamcontrol_punch_gui_preflight.log"
mkdir -p "$MON"
[[ "$(find "$REF" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 100 ]] || {
  echo 'BLOCKED punch_references_missing' > "$STATE"; exit 1;
}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
echo 'STARTING task=Punch envs=64 max_iterations=1000 gpu=0 livestream=2 nonbaseline=true' > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 1800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Punch-v0 \
  --device=cuda:0 --seed=42 --num_envs=64 --max_iterations=1000 \
  --livestream=2 > "$LOG" 2>&1
rc=$?
set -e
echo "STOPPED exit=$rc nonbaseline=true log=$LOG" > "$STATE"
