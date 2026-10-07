#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:?collect or evaluate}"
SEED="${2:?42/43/44}"
ART="$ROOT/artifacts/world_confirmation_20261007"
PREFIX="$ROOT/artifacts/monitoring/r3_world_confirmation_${MODE}_${SEED}_v1"
OUT="$ART/${MODE}_seed${SEED}_v1"
[[ ! -e "$OUT" && ! -e "$PREFIX.state" ]] || { echo 'Preserve existing invocation'; exit 1; }
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo 'GPU0_BUSY'; exit 1; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DREAMCONTROL_KICK_REF_PATH="$ART/selection/selected_refs"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils:$ROOT/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo "RUNNING mode=$MODE seed=$SEED PPO=false auxiliary_updates=0" > "$PREFIX.state"
set +e
if [[ "$MODE" == collect ]]; then
 timeout --signal=TERM --kill-after=30s 1800 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_ppo_isaac.py" --mode collect --output "$OUT" --seed "$SEED" --num_envs 2048 \
 --rollouts 20 --device cuda:0 --headless --disturbance lateral80 --monitor-prefix "$PREFIX" \
 --reference-manifest "$ART/selection/manifest.json" > "$PREFIX.log" 2>&1
 rc=$?
 SENTINEL=WORLD_PPO_ISAAC_COMPLETE
elif [[ "$MODE" == evaluate ]]; then
 timeout --signal=TERM --kill-after=30s 900 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
 "$ROOT/scripts/r3_world_confirmation_eval_20261007.py" --data "$ART/collect_seed${SEED}_v1" \
 --output "$OUT" --seed "$SEED" --device cuda:0 --model-lock "$ART/model_lock.json" > "$PREFIX.log" 2>&1
 rc=$?
 SENTINEL=WORLD_RESIDUAL_NEW_REFERENCE_CONFIRMATION_COMPLETE
else
 echo "FAILED invalid mode=$MODE" > "$PREFIX.state"; exit 2
fi
set -e
if (( rc )) || ! grep -q "$SENTINEL" "$PREFIX.log"; then
 echo "FAILED mode=$MODE exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo "COMPLETE mode=$MODE exit=0 PPO=false auxiliary_updates=0" > "$PREFIX.state"
echo "WORLD_CONFIRMATION_DOCUMENTED_STAGE_COMPLETE $MODE $SEED"
