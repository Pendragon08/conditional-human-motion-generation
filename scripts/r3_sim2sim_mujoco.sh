#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python
MODE="${1:-witness}"
TASK="${2:-Kick}"
cd "$ROOT"
"$PY" -u scripts/r3_sim2sim_mujoco.py --mode "$MODE" --task "$TASK" "${@:3}"
