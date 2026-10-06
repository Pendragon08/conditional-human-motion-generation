#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:?capture or smoke}"
case "$MODE" in capture|smoke) ;; *) exit 2 ;; esac
PREFIX="$ROOT/artifacts/monitoring/r3_feedback_isaac_continuation_20261006_$MODE"
OUT="$ROOT/artifacts/feedback_isaac_20261006/$MODE"
[[ ! -e "$OUT" ]] || { echo 'Existing invocation must be preserved'; exit 1; }
mkdir -p "$ROOT/artifacts/monitoring"
exec 9>"$ROOT/artifacts/monitoring/r3_feedback_isaac_gpu0_20261006.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo "RUNNING mode=$MODE frozen_policy=true ppo_updates=0" > "$PREFIX.state"
set +e
timeout --signal=TERM --kill-after=30s 900s "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_feedback_isaac_continuation_20261006.py" \
  --mode "$MODE" --output "$OUT" --steps 500 --device cuda:0 --headless > "$PREFIX.log" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q ISAAC_FEEDBACK_CONTINUATION_COMPLETE "$PREFIX.log"; then
  echo "FAILED mode=$MODE exit=$rc preserve_log=true" > "$PREFIX.state"; exit 1
fi
echo "COMPLETE mode=$MODE exit=0 physical_frames=500 formal_feedback_gate=false" > "$PREFIX.state"
echo "ISAAC_FEEDBACK_DOCUMENTED_WITNESS_COMPLETE mode=$MODE"
