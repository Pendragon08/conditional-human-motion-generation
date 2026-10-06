#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec 9>artifacts/monitoring/r3_feedback_world_learned_control_20261006.lock
flock -n 9 || exit 1
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u scripts/r3_feedback_world_learned_control_20261006.py --stage "${1:?stage required}"
