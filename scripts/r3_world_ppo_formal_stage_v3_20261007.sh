#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:?train/eval/eval-diagnostic}";ARM="${2:?arm}";SEED="${3:?seed}"
[[ "$ARM" =~ ^(P0|P-cap|P-history|W-correct|W-shuffled-original)$ ]] || exit 2
[[ "$SEED" =~ ^(42|43|44)$ ]] || exit 2
VERSION=v1
BASE="$ROOT/artifacts/world_residual_ppo_20261007/formal"
if [[ "$MODE" == eval-diagnostic ]]; then BASE="$ROOT/artifacts/world_residual_ppo_20261007/eval_interface_v3"; fi
PREFIX="$ROOT/artifacts/monitoring/r3_world_ppo_${MODE}_${ARM}_${SEED}_20261007"
if [[ "$MODE" == eval-diagnostic ]]; then PREFIX="${PREFIX}_v3"; fi
OUT="$BASE/${MODE}_${ARM}_seed${SEED}_$VERSION"
[[ ! -e "$PREFIX.state" && ! -e "$OUT" ]] || { echo PRESERVE_EXISTING_INVOCATION; exit 1; }
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo GPU0_BUSY; exit 1; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DREAMCONTROL_KICK_REF_PATH="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils:$ROOT/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$ROOT/third_party/DreamControl/Training"
echo "RUNNING mode=$MODE arm=$ARM seed=$SEED" > "$PREFIX.state"
set +e
if [[ "$MODE" == train ]]; then
 timeout --signal=TERM --kill-after=60s 86400 "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_world_ppo_control_formal_v3_20261007.py" --mode formal --arm "$ARM" --seed "$SEED" \
  --output "$OUT" --num_envs 8192 --updates 500 --aux-updates 1 --aux-batch 1024 \
  --monitor-prefix "$PREFIX" --device cuda:0 --headless > "$PREFIX.log" 2>&1
 rc=$?
 SENTINEL=WORLD_RESIDUAL_PPO_CONTROL_COMPLETE
elif [[ "$MODE" == eval || "$MODE" == eval-diagnostic ]]; then
 if [[ "$MODE" == eval ]]; then
  CHECKPOINT="$BASE/train_${ARM}_seed${SEED}_v1/model_update0500.pt";ENVS=2048;STEPS=500;EVALMODE=formal
 else
  CHECKPOINT="$ROOT/artifacts/world_residual_ppo_20261007/capacity_${ARM}_seed42_v2/model_update0002.pt";ENVS=64;STEPS=120;EVALMODE=diagnostic
 fi
 export DREAMCONTROL_KICK_REF_PATH="$ROOT/artifacts/world_confirmation_20261007/selection/selected_refs"
 LIMIT=1800; if [[ "$MODE" == eval-diagnostic ]]; then LIMIT=300; fi
 timeout --signal=TERM --kill-after=30s "$LIMIT" "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_world_ppo_eval_isaac_v3_20261007.py" --mode "$EVALMODE" --checkpoint "$CHECKPOINT" \
  --output "$OUT" --seed "$SEED" --num_envs "$ENVS" --steps "$STEPS" \
  --reference-manifest "$ROOT/artifacts/world_confirmation_20261007/selection/manifest.json" \
  --monitor-prefix "$PREFIX" --device cuda:0 --headless > "$PREFIX.log" 2>&1
 rc=$?
 SENTINEL=WORLD_PPO_CONTROL_EVALUATION_COMPLETE
else
 rc=2;SENTINEL=INVALID
fi
set -e
if (( rc )) || ! grep -q "$SENTINEL" "$PREFIX.log"; then
 echo "FAILED mode=$MODE arm=$ARM seed=$SEED child_exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo "COMPLETE mode=$MODE arm=$ARM seed=$SEED child_exit=0" > "$PREFIX.state"
echo WORLD_PPO_FORMAL_STAGE_COMPLETE
