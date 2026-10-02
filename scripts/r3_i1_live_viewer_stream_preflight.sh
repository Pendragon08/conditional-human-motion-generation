#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_live_viewer_stream_preflight.state"
LOG="$MON/r3_i1_live_viewer_stream_preflight.log"
mkdir -p "$MON" "$ROOT/cache/r3_i1_viewer_stream"
exec 9>"$MON/r3_i1_live_viewer_stream_preflight.lock"
flock -n 9 || exit 1
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export PUBLIC_IP=10.147.17.32
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled
export TMPDIR="$ROOT/cache/r3_i1_viewer_stream/tmp"
export XDG_CACHE_HOME="$ROOT/cache/r3_i1_viewer_stream/xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_i1_viewer_stream/mpl"
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo 'STARTING mode=synthetic_self_test_not_training gpu=0 webrtc=2' > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=30s 900s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_i1_live_training_viewer.py" \
  --device=cuda:0 --livestream=2 --timeout-seconds=600 --self-test \
  --kit_args="--portable-root $ROOT/cache/r3_i1_viewer_stream/kit --/renderer/multiGpu/enabled=false" \
  --status-path="$STATE" \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'R3_VIEWER_APPLIED_POSE mode=viewer_self_test' "$LOG" || grep -q 'Traceback (most recent call last)' "$LOG"; then
  echo "FAILED exit=$rc mode=synthetic_self_test_not_training log=$LOG" > "$STATE"
  exit 1
fi
echo 'COMPLETE mode=synthetic_self_test_not_training training_unaffected=true' > "$STATE"
