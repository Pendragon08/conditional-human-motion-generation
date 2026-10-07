#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/artifacts/monitoring/r3_world_residual_visibility_20261007"
OUT="$ROOT/artifacts/world_residual_ppo_20261007/visibility_seed42_v1"
[[ ! -e "$PREFIX.state" && ! -e "$OUT" ]] || { echo 'Preserve existing invocation'; exit 1; }
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo 'GPU0_BUSY'; exit 1; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils:$ROOT/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo 'RUNNING mode=visibility formal=false frozen_policy=true' > "$PREFIX.state"
set +e
timeout --signal=TERM --kill-after=30s 1500 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_confirmation_isaac_v2_20261007.py" --mode visibility --output "$OUT" \
 --seed 42 --num_envs 16 --visibility-seconds 1200 --monitor-prefix "$PREFIX" --device cuda:0 --headless \
 > "$PREFIX.log" 2>&1
rc=$?
set -e
if (( rc )) || ! grep -q WORLD_PPO_ISAAC_COMPLETE "$PREFIX.log"; then
 echo "FAILED mode=visibility child_exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo 'COMPLETE mode=visibility child_exit=0 formal=false' > "$PREFIX.state"
