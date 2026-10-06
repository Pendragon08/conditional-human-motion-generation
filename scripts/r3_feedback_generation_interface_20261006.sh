#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
ARM="${1:?arm required}"
case "$ARM" in predicted|actual|shuffled) ;; *) exit 2 ;; esac
OUT="$ROOT/artifacts/feedback_world_model_20261006/generation_v3/$ARM"
[[ ! -e "$OUT" ]] || { echo "Existing arm must be preserved: $OUT" >&2; exit 1; }
mkdir -p "$ROOT/artifacts/monitoring"
exec 9>"$ROOT/artifacts/monitoring/r3_feedback_generation_interface_20261006.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$TRAJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$TRAJ"
timeout --signal=TERM --kill-after=30s 600s "$OLD/.venv-pbhc-training/bin/python" -u \
  "$ROOT/scripts/r3_feedback_generation_interface_pilot.py" --stage generate --arm "$ARM" \
  --seed 42 --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt \
  --num_repetitions 1 --task kick --output_dir "$OUT" --batch_size 3
"$OLD/.venv-pbhc-training/bin/python" - "$OUT/results.npy" <<'PY'
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True).item()
assert d['motion'].shape==(3,22,3,196) and np.isfinite(d['motion']).all()
print('FINITE_GENERATION_WITNESS',d['motion'].shape)
PY
echo "FEEDBACK_GENERATION_ARM_COMPLETE arm=$ARM"
