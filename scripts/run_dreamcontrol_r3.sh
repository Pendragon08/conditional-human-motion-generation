#!/usr/bin/env bash
set -euo pipefail

# Independent R3 baseline; leave historical PBHC training environments untouched.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
ITERATIONS="${1:?iterations required}"
NUM_ENVS="${2:?num_envs required}"
MODE="${3:-smoke}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp"
export PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true
export HYDRA_FULL_ERROR=1
export PUBLIC_IP=10.147.17.32
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$ROOT/artifacts/monitoring"

cd "$TRAIN"
echo "R3_DREAMCONTROL_START mode=$MODE iterations=$ITERATIONS num_envs=$NUM_ENVS seed=123 gpu=cuda:0 livestream=2"
echo "R3_DREAMCONTROL_REFERENCE=$(realpath ../TrajGen/sample/Kick_sim2)"
if [[ "$MODE" == smoke ]]; then
  VIDEO_INTERVAL=1
  VIDEO_LENGTH=24
else
  VIDEO_INTERVAL=2000
  VIDEO_LENGTH=200
fi

python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
"$ROOT/third_party/isaacsim-4.5.0/python.sh" -u "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 \
  --seed=123 \
  --num_envs="$NUM_ENVS" \
  --max_iterations="$ITERATIONS" \
  --livestream=2 \
  --video \
  --video_length="$VIDEO_LENGTH" \
  --video_interval="$VIDEO_INTERVAL"

echo "R3_DREAMCONTROL_PROCESS_RETURNED mode=$MODE (verify training artifacts before accepting)"
