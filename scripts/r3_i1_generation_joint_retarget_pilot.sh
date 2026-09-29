#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_generation_joint_retarget_pilot.state"
BASE="$ROOT/artifacts/r3_i1_generation_joint_pilot"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_generation_joint_retarget_pilot.lock"
flock -n 9 || { echo 'Retarget pilot already active' >&2; exit 1; }
grep -q '^COMPLETE arms=baseline,joint' "$MON/r3_i1_generation_joint_pilot.state" || {
  echo 'BLOCKED generation_incomplete' > "$STATE"; exit 1;
}
"$OLD/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_i1_generation_joint_compare.py" \
  --base "$BASE" --output "$ROOT/results/r3_i1_generation_joint_human_pilot.json" \
  > "$MON/r3_i1_generation_joint_human_compare.log" 2>&1 || {
  echo 'FAILED human_compare' > "$STATE"; exit 1;
}
for arm in baseline joint; do
  STAGE="$TRAJ/sample_i1_generation_joint_pilot_$arm"
  [[ ! -e "$STAGE" ]] || { echo "BLOCKED stage_exists=$STAGE" > "$STATE"; exit 1; }
  mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1"
  cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
  cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
  ln -s "$BASE/$arm/results.npy" "$STAGE/Kick_sim/results.npy"
  export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  echo "RUNNING arm=$arm original_retarget=true no_extra_ik=true" > "$STATE"
  set +e
  (cd "$STAGE/Kick_sim" && timeout --signal=TERM --kill-after=30s 1800s \
      "$OLD/.venv-retarget/bin/python" -u retarget.py) \
      > "$MON/r3_i1_generation_joint_retarget_${arm}.log" 2>&1
  rc=$?
  set -e
  count="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
  if (( rc != 0 || count < 1 )); then
    echo "FAILED arm=$arm exit=$rc accepted=$count" > "$STATE"; exit 1
  fi
done
echo 'COMPLETE arms=baseline,joint original_retarget=true no_extra_ik=true' > "$STATE"
