#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$ROOT"
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u \
  scripts/r3_mujoco_robust_contact.py "$@"
