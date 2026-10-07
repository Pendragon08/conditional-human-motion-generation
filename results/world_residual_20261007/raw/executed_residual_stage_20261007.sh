#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:?probe, collect, fit or visibility}"
SEED="${2:-42}"
RUNID="${3:-v1}"
[[ "$RUNID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
PREFIX="$ROOT/artifacts/monitoring/r3_world_residual_${MODE}_${SEED}_${RUNID}"
OUT="$ROOT/artifacts/world_residual_20261007/${MODE}_seed${SEED}_${RUNID}"
[[ ! -e "$OUT" && ! -e "$PREFIX.state" ]] || { echo 'Preserve existing invocation'; exit 1; }
mkdir -p "$ROOT/artifacts/monitoring" "$ROOT/artifacts/world_residual_20261007"
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo 'WORLD_PPO_GPU0_LOCK_BUSY'; exit 1; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils:$ROOT/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo "RUNNING mode=$MODE seed=$SEED id=$RUNID formal=false" > "$PREFIX.state"
case "$MODE" in
 probe) N=16; LIMIT=900; ISAAC_MODE=collect;;
 collect) N=8192; LIMIT=7200; ISAAC_MODE=collect;;
 visibility) N=16; LIMIT=1500; ISAAC_MODE=visibility;;
 fit) LIMIT=7200;;
 *) exit 2;;
esac
set +e
if [[ "$MODE" == fit ]]; then
 timeout --signal=TERM --kill-after=30s "$LIMIT" "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_ppo_residual_fit_20261007.py" \
 --data "$ROOT/artifacts/world_residual_20261007/collect_seed${SEED}_v1" \
 --output "$OUT" --seed "$SEED" --device cuda:0 > "$PREFIX.log" 2>&1
 SENTINEL=WORLD_RESIDUAL_PREDICTION_COMPLETE
else
 timeout --signal=TERM --kill-after=30s "$LIMIT" "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_ppo_isaac.py" --mode "$ISAAC_MODE" --output "$OUT" --seed "$SEED" \
 --num_envs "$N" --rollouts 20 --device cuda:0 --headless --disturbance lateral80 \
 --monitor-prefix "$PREFIX" > "$PREFIX.log" 2>&1
 SENTINEL=WORLD_PPO_ISAAC_COMPLETE
fi
rc=$?
set -e
if (( rc != 0 )) || ! grep -q "$SENTINEL" "$PREFIX.log"; then
 echo "FAILED mode=$MODE exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo "COMPLETE mode=$MODE exit=0 formal=false" > "$PREFIX.state"
echo "WORLD_RESIDUAL_DOCUMENTED_STAGE_COMPLETE mode=$MODE seed=$SEED id=$RUNID"
