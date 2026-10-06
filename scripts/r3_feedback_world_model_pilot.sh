#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p artifacts/monitoring
case "${1:?stage required}" in
  witness|fit_witness) LOCK=artifacts/monitoring/r3_feedback_world_model_witness_20261006.lock ;;
  *) LOCK=artifacts/monitoring/r3_feedback_world_model_pilot_20261006.lock ;;
esac
exec 9>"$LOCK"
flock -n 9 || { echo 'Pilot lock already held' >&2; exit 1; }
"$OLD/.venv-pbhc-training/bin/python" -u scripts/r3_feedback_world_model_pilot.py --stage "${1:?stage required}"
