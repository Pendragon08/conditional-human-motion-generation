#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PARENT="${PHYGILE_PARENT:-/data/code/ouyanghao/oyh/phygile-reproduction-20261017}"
SHARED=/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party
export ISAAC_SIM_ROOT="$SHARED/isaacsim-4.5.0" ISAAC_LAB_ROOT="$SHARED/IsaacLab"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled WANDB_SILENT=true
export G1_TRAINING_CFG_PATH="$PARENT/configs/moe_pilot.yaml"
export XDG_CACHE_HOME="$PARENT/cache/xdg" MPLCONFIGDIR="$PWD/cache/matplotlib" TMPDIR="$PWD/cache/tmp"
export PHYGILE_USD_CACHE="$PARENT/cache/usd/g1" PHYGILE_OFFLINE_ASSETS=1
export PYTHONPATH="$PWD/scripts:$PWD/vendor/unitree_rl_lab/source/unitree_rl_lab:$PARENT/upstream/source/whole_body_tracking:$ISAAC_LAB_ROOT/source/isaaclab:$ISAAC_LAB_ROOT/source/isaaclab_assets:$ISAAC_LAB_ROOT/source/isaaclab_tasks:$ISAAC_LAB_ROOT/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec "$ISAAC_SIM_ROOT/python.sh" -u "$@"
