#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
OUT="$ROOT/results/condition_feedback_formal_v28_20261012"
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_SCENE_UDP R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
PY="$RUNTIME/.venv-pbhc-training/bin/python"
cd "$ROOT/third_party/CLoSD"
test "$(git rev-parse HEAD)" = d8950bdb78312fcd2844f336b3fc079e7368c87f
test -z "$(git diff --name-only HEAD -- closd)"
test "${1:-}" = collect
shift
exec "$PY" -u "$ROOT/scripts/contact_com_collect_v28_20261012.py" --manifest "$OUT/source_manifest.json" "$@"
