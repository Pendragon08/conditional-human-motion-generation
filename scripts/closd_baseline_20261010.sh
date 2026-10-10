#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
OUT="$ROOT/results/closd_baseline_20261010"
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
case "${1:-}" in
  --check)
    "$PY" "$ROOT/scripts/r3_closd_env_witness.py"
    "$PY" "$ROOT/scripts/closd_t2m_record_20261010.py" --output "$OUT/smoke" --envs 4 --episodes 4 --seed 10
    ;;
  --run)
    exec 9>"$OUT/pipeline.lock"
    flock -n 9 || { echo 'Another baseline run owns the lock'; exit 1; }
    test ! -e "$OUT/pipeline.state"
    trap 'rc=$?; if ((rc)); then echo "FAILED exit=$rc" > "$OUT/pipeline.state"; fi' EXIT
    echo 'RUNNING released_motion_evaluation' > "$OUT/pipeline.state"
    mkdir -p "$OUT/released_evaluation"
    ln -s "$(realpath closd/diffusion_planner/saved_motions/closd/CloSD.pkl)" "$OUT/released_evaluation/CloSD.pkl"
    timeout --signal=TERM --kill-after=30 7200 "$PY" -m closd.diffusion_planner.eval.eval_humanml \
      --external_results_file "$OUT/released_evaluation/CloSD.pkl" --do_unique --seed 10 \
      > "$OUT/released_evaluation/stdout.log" 2>&1
    echo 'RUNNING fresh_20000_physics_episodes' > "$OUT/pipeline.state"
    timeout --signal=TERM --kill-after=45 14400 "$PY" -u "$ROOT/scripts/closd_t2m_record_20261010.py" \
      --output "$OUT/fresh_generation" --envs 4096 --episodes 20000 --seed 10 \
      > "$OUT/fresh_generation.log" 2>&1
    echo 'RUNNING fresh_motion_evaluation' > "$OUT/pipeline.state"
    timeout --signal=TERM --kill-after=30 7200 "$PY" -m closd.diffusion_planner.eval.eval_humanml \
      --external_results_file "$OUT/fresh_generation/CLoSD.pkl" --do_unique --seed 10 \
      > "$OUT/fresh_evaluation.log" 2>&1
    echo 'COMPLETE released_eval_and_fresh_generation_and_eval' > "$OUT/pipeline.state"
    ;;
  *) echo 'Use --check or --run'; exit 2;;
esac
