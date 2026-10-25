#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024
SHARED_ROOT=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party
export ISAAC_SIM_ROOT="$SHARED_ROOT/isaacsim-4.5.0" ISAAC_LAB_ROOT="$SHARED_ROOT/IsaacLab"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled WANDB_SILENT=true
export XDG_CACHE_HOME="$TASK_ROOT/cache/tracker/xdg" MPLCONFIGDIR="$TASK_ROOT/cache/tracker/matplotlib" TMPDIR="$TASK_ROOT/cache/tracker/tmp"
export PYTHONPATH="$TASK_ROOT/scripts:$TASK_ROOT/upstream/TextOpTracker/source/textop_tracker:$ISAAC_LAB_ROOT/source/isaaclab:$ISAAC_LAB_ROOT/source/isaaclab_assets:$ISAAC_LAB_ROOT/source/isaaclab_tasks:$ISAAC_LAB_ROOT/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
cd "$TASK_ROOT"
exec "$ISAAC_SIM_ROOT/python.sh" -u "$@"
