#!/usr/bin/env bash
set -uo pipefail

# Recovery of the stalled R3 Kick baseline. Keep the original run immutable.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
OLD_RUN=2026-09-25_03-56-03-Kick-a
OLD_CKPT="$TRAIN/logs/rsl_rl/g1/$OLD_RUN/model_1300.pt"
PREFIX="$ROOT/artifacts/monitoring/r3_dreamcontrol_kick_resume1300_700_1024"
LOG="$PREFIX.log"
EXIT="$PREFIX.exit"
START="$PREFIX.start"

if [[ -e "$START" || -e "$EXIT" ]]; then
  echo "Recovery launch already attempted: $PREFIX" >&2
  exit 2
fi
if [[ ! -s "$OLD_CKPT" ]]; then
  echo "Missing source checkpoint: $OLD_CKPT" >&2
  exit 3
fi
mkdir -p "$ROOT/artifacts/monitoring"
touch "$START"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PUBLIC_IP=10.147.17.32
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$TRAIN"
{
  echo "R3_RESUME_START source=$OLD_CKPT seed=123 gpu=cuda:0 num_envs=1024 additional_iterations=700 target_final_iteration=1999 livestream=2"
  python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
  timeout --signal=TERM --kill-after=45s 172800s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 \
    --device=cuda:0 --seed=123 --num_envs=1024 \
    --resume --load_run="$OLD_RUN" --checkpoint=model_1300.pt \
    --max_iterations=700 --livestream=2 \
    --video --video_length=200 --video_interval=2000
} > "$LOG" 2>&1
result=$?
if grep -Eq 'Error executing job|ModuleNotFoundError|Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
  result=90
fi
if (( result == 0 )) && ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  result=91
fi
printf '%s\n' "$result" > "$EXIT"
echo "R3_RESUME_EXIT=$result" >> "$LOG"
exit "$result"
