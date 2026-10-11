#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_SCENE_UDP R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
PY="$RUNTIME/.venv-pbhc-training/bin/python"
OUT="$ROOT/results/physics_guided_20261011"
cd "$ROOT/third_party/CLoSD"
test "$(git rev-parse HEAD)" = d8950bdb78312fcd2844f336b3fc079e7368c87f
test -z "$(git diff --name-only HEAD -- closd)"
case "${1:-}" in
  check)
    "$PY" "$ROOT/scripts/physics_guided_core.py"
    "$PY" "$ROOT/scripts/physics_guided_measurement_checks.py"
    "$PY" "$ROOT/scripts/r3_closd_env_witness.py"
    timeout --signal=TERM --kill-after=30 1200 "$PY" -u "$ROOT/scripts/physics_guided_collect.py" \
      --output "$OUT/${2:-witness}" --manifest "$OUT/source_manifest.json" --limit-sources 2 --seed 11 --mode planned
    ;;
  collect)
    shift
    exec "$PY" -u "$ROOT/scripts/physics_guided_collect.py" --manifest "$OUT/source_manifest.json" "$@"
    ;;
  queue)
    exec "$PY" -u "$ROOT/scripts/physics_guided_queue.py" --output "$OUT"
    ;;
  *) echo 'Use check [fresh-name], collect [args], or queue'; exit 2;;
esac
