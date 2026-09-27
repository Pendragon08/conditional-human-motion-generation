#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY="$ROOT/envs/closd-r3/site-packages"
export PYTHONPATH="$OVERLAY:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$OVERLAY/ninja/data/bin:$OVERLAY/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
cd "$ROOT"
timeout 180 "$ROOT/.venv-pbhc-training/bin/python" third_party/CLoSD/closd/run.py env=closd_multitask --help
