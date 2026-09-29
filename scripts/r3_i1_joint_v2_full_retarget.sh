#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_i1_joint_v2_full_retarget.state"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_joint_v2_full_retarget.lock"
flock -n 9 || { echo 'Full retarget already active' >&2; exit 1; }
echo 'WAITING_FOR_GENERATION' > "$STATE"
for _ in $(seq 1 600); do
  if grep -q '^COMPLETE seeds=42,43,44' "$MON/r3_i1_joint_v2_full_generation.state" 2>/dev/null; then break; fi
  if grep -q '^FAILED\|^BLOCKED' "$MON/r3_i1_joint_v2_full_generation.state" 2>/dev/null; then
    echo 'BLOCKED generation_failed' > "$STATE"; exit 1
  fi
  sleep 20
done
grep -q '^COMPLETE seeds=42,43,44' "$MON/r3_i1_joint_v2_full_generation.state" || {
  echo 'BLOCKED generation_wait_timeout' > "$STATE"; exit 1;
}
for seed in 42 43 44; do
  STAGE="$TRAJ/sample_i1_joint_v2_seed$seed"
  SOURCE="$ROOT/artifacts/i1_joint_v2_candidates/seed$seed/results.npy"
  BASELINE="$ROOT/artifacts/i1_fixed_candidates/seed$seed/results.npy"
  [[ -s "$SOURCE" && -s "$BASELINE" && ! -e "$STAGE" ]] || {
    echo "BLOCKED seed=$seed source_missing_or_stage_exists" > "$STATE"; exit 1;
  }
  mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1"
  cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
  cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
  ln -s "$SOURCE" "$STAGE/Kick_sim/results.npy"
  export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  echo "RUNNING seed=$seed original_retarget=true no_extra_ik=true" > "$STATE"
  set +e
  (cd "$STAGE/Kick_sim" && timeout --signal=TERM --kill-after=30s 3600s \
      "$OLD/.venv-retarget/bin/python" -u retarget.py) \
      > "$MON/r3_i1_joint_v2_retarget_seed$seed.log" 2>&1
  rc=$?
  set -e
  count="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
  if (( rc != 0 || count < 1 )); then
    echo "FAILED seed=$seed exit=$rc accepted=$count" > "$STATE"; exit 1
  fi
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    "$OLD/.venv-retarget/bin/python" "$ROOT/scripts/r3_i1_constraint_transfer_audit.py" \
    --stage "$STAGE" --constraint-source "$BASELINE" \
    --output "$ROOT/results/r3_i1_joint_v2_robot_seed$seed.json" \
    > "$MON/r3_i1_joint_v2_robot_audit_seed$seed.log" 2>&1 || {
    echo "FAILED seed=$seed robot_audit" > "$STATE"; exit 1;
  }
done
echo 'COMPLETE seeds=42,43,44 original_retarget=true no_extra_ik=true' > "$STATE"
