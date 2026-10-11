#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
OUT="$ROOT/results/gap_diagnostic_20261011/v3"
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_SCENE_UDP R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
PY="$RUNTIME/.venv-pbhc-training/bin/python"
mkdir -p "$OUT"
cd "$ROOT/third_party/CLoSD"
test "$(git rev-parse HEAD)" = d8950bdb78312fcd2844f336b3fc079e7368c87f
test -z "$(git diff --name-only HEAD -- closd)"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'Diagnostic already owns lock'; exit 1; }
case "${1:-}" in
  --check)
    "$PY" "$ROOT/scripts/r3_closd_env_witness.py"
    timeout --signal=TERM --kill-after=30 1200 "$PY" -u "$ROOT/scripts/closd_gap_diagnostic_20261011.py" --output "$OUT/witness" --envs 4 --seed 11
    ;;
  --run)
    test ! -e "$OUT/run.state"
    trap 'rc=$?; if ((rc)); then echo "FAILED exit=$rc" > "$OUT/run.state"; fi' EXIT
    for seed in 11 12 13; do
      echo "RUNNING seed=$seed" > "$OUT/run.state"
      timeout --signal=TERM --kill-after=30 1800 "$PY" -u "$ROOT/scripts/closd_gap_diagnostic_20261011.py" --output "$OUT/seed_$seed" --envs 64 --seed "$seed" > "$OUT/seed_$seed.log" 2>&1
    done
    echo 'COLLECTION_COMPLETE 3x64 first attempts' > "$OUT/run.state"
    ;;
  *) echo 'Use --check or --run'; exit 2;;
esac
