#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SHARED_ROOT="${PHYGILE_SHARED_ROOT:-/data/code/ouyanghao/oyh/humanoid-combat-motion}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$SHARED_ROOT/third_party/isaacsim-4.5.0}"
export ISAAC_LAB_ROOT="${ISAAC_LAB_ROOT:-$SHARED_ROOT/third_party/IsaacLab}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=disabled WANDB_SILENT=true
export XDG_CACHE_HOME="$PWD/cache/xdg" MPLCONFIGDIR="$PWD/cache/matplotlib" TMPDIR="$PWD/cache/tmp"
export PHYGILE_USD_CACHE="$PWD/cache/usd/g1" PHYGILE_OFFLINE_ASSETS=1
export PYTHONPATH="$PWD/upstream/source/whole_body_tracking:$ISAAC_LAB_ROOT/source/isaaclab:$ISAAC_LAB_ROOT/source/isaaclab_assets:$ISAAC_LAB_ROOT/source/isaaclab_tasks:$ISAAC_LAB_ROOT/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PHYGILE_USD_CACHE"
exec "$ISAAC_SIM_ROOT/python.sh" -u "$@"
