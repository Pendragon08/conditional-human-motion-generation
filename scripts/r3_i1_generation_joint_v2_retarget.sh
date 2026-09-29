#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
STAGE="$TRAJ/sample_i1_generation_joint_pilot_joint_v2"
BASE="$ROOT/artifacts/r3_i1_generation_joint_pilot"
STATE="$MON/r3_i1_generation_joint_v2_retarget.state"
mkdir -p "$MON"
exec 9>"$MON/r3_i1_generation_joint_v2_retarget.lock"
flock -n 9 || { echo 'Already running' >&2; exit 1; }
[[ -s "$BASE/joint_v2/results.npy" && ! -e "$STAGE" ]] || {
  echo 'BLOCKED source_missing_or_stage_exists' > "$STATE"; exit 1;
}
mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1"
cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
ln -s "$BASE/joint_v2/results.npy" "$STAGE/Kick_sim/results.npy"
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo 'RUNNING original_retarget=true no_extra_ik=true arm=joint_v2' > "$STATE"
set +e
(cd "$STAGE/Kick_sim" && timeout --signal=TERM --kill-after=30s 1800s \
    "$OLD/.venv-retarget/bin/python" -u retarget.py) \
    > "$MON/r3_i1_generation_joint_v2_retarget.log" 2>&1
rc=$?
set -e
count="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
if (( rc != 0 || count < 1 )); then
  echo "FAILED exit=$rc accepted=$count" > "$STATE"; exit 1
fi
echo "COMPLETE accepted=$count/8 original_retarget=true no_extra_ik=true" > "$STATE"
