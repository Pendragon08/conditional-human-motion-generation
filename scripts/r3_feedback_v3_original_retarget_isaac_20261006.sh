#!/usr/bin/env bash
# Isolated execution of the pinned official retarget/refine stages. No added IK.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
ARM="${1:?predicted, actual, or shuffled}"
case "$ARM" in predicted|actual|shuffled) ;; *) exit 2 ;; esac
PREFIX="$ROOT/artifacts/monitoring/r3_feedback_v3_original_retarget_isaac_20261006_$ARM"
STAGE="$TRAJ/sample_feedback_v3_isaac_20261006_$ARM"
SOURCE="$ROOT/artifacts/feedback_world_model_20261006/generation_v3/$ARM/results.npy"
mkdir -p "$ROOT/artifacts/monitoring"
exec 9>"$ROOT/artifacts/monitoring/r3_feedback_v3_original_retarget_isaac_20261006.lock"
flock -n 9 || { echo 'Other original retarget stage active'; exit 1; }
[[ -s "$SOURCE" && ! -e "$STAGE" ]] || { echo 'Source missing or stage already exists; preserve it'; exit 1; }
[[ -x "$OLD/.venv-retarget/bin/python" ]] || exit 1
mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1" "$STAGE/Kick_sim2"
cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
cp "$TRAJ/sample/Kick_sim1/refine_motions.py" "$STAGE/Kick_sim1/refine_motions.py"
cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
ln -s "$SOURCE" "$STAGE/Kick_sim/results.npy"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "RUNNING arm=$ARM stage=original_retarget no_extra_ik=true" > "$PREFIX.state"
set +e
(cd "$STAGE/Kick_sim" && timeout --signal=TERM --kill-after=30s 1800s "$OLD/.venv-retarget/bin/python" -u retarget.py) > "$PREFIX.log" 2>&1
rc=$?
set -e
count="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
if (( rc != 0 || count == 0 )); then
  echo "FAILED arm=$ARM exit=$rc accepted=$count denominator=3" > "$PREFIX.state"; exit 1
fi
echo "RUNNING arm=$ARM stage=official_refinement accepted=$count denominator=3" > "$PREFIX.state"
set +e
(cd "$STAGE/Kick_sim1" && timeout --signal=TERM --kill-after=30s 600s "$OLD/.venv-retarget/bin/python" -u refine_motions.py) >> "$PREFIX.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then echo "FAILED arm=$ARM stage=refinement exit=$rc" > "$PREFIX.state"; exit 1; fi
echo "COMPLETE arm=$ARM accepted=$count denominator=3 no_extra_ik=true formal_control_gate=false" > "$PREFIX.state"
echo "FEEDBACK_ORIGINAL_RETARGET_ISAAC_READY arm=$ARM accepted=$count denominator=3"
