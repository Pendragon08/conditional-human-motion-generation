#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/artifacts/monitoring/r3_world_ppo_capacity_20261007"
[[ ! -e "$PREFIX.state" ]] || exit 1
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo GPU0_BUSY; exit 1; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils:$ROOT/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo 'RUNNING stage=CUDA_fixture formal=false' > "$PREFIX.state"
set +e
timeout --signal=TERM --kill-after=30s 120 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_ppo_invariant_witness_20261007.py" \
 --output "$ROOT/artifacts/world_residual_ppo_20261007/invariant_fixture_v1" > "$PREFIX.fixture.log" 2>&1
rc=$?
set -e
if (( rc )) || ! grep -q WORLD_PPO_INVARIANT_FIXTURE_COMPLETE "$PREFIX.fixture.log"; then
 echo "FAILED stage=fixture child_exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo 'COMPLETE child_exit=0 PPO=0 simulation=0' > "$PREFIX.fixture.state"
for ARM in P-history W-correct; do
 OUT="$ROOT/artifacts/world_residual_ppo_20261007/capacity_${ARM}_seed42_v2"
 [[ ! -e "$OUT" ]] || exit 1
 echo "RUNNING stage=capacity arm=$ARM formal=false num_envs=8192" > "$PREFIX.state"
 set +e
 timeout --signal=TERM --kill-after=30s 1800 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_world_ppo_control_capacity_v2_20261007.py" --mode capacity --arm "$ARM" --seed 42 \
  --output "$OUT" --num_envs 8192 --updates 2 --aux-updates 1 --aux-batch 1024 \
  --monitor-prefix "$PREFIX.$ARM" --device cuda:0 --headless > "$PREFIX.$ARM.log" 2>&1
 rc=$?
 set -e
 if (( rc )) || ! grep -q WORLD_RESIDUAL_PPO_CONTROL_COMPLETE "$PREFIX.$ARM.log"; then
  echo "FAILED stage=capacity arm=$ARM child_exit=$rc preserve=true" > "$PREFIX.state"; exit 1
 fi
 echo "COMPLETE arm=$ARM child_exit=0 formal=false num_envs=8192 diagnostic_updates=2" > "$PREFIX.$ARM.state"
done
echo 'COMPLETE fixture_exit=0 P-history_exit=0 W-correct_exit=0 formal=false' > "$PREFIX.state"
echo WORLD_RESIDUAL_PPO_CAPACITY_QUEUE_COMPLETE
