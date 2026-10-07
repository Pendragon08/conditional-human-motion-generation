#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/data/code/ouyanghao/oyh/humanoid-combat-motion
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
STAGE="${1:?stage}"; SEED="${2:?seed}"; OUT="${3:?output}"; ARM="${4:-}"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
case "$STAGE" in
physics|witness)
 export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$ROOT/third_party/DreamControl/Training/source/isaaclab:$ROOT/third_party/DreamControl/Training/source/isaaclab_assets:$ROOT/third_party/DreamControl/Training/source/isaaclab_tasks:$ROOT/third_party/DreamControl/Training/source/isaaclab_rl:$ROOT/third_party/DreamControl/Training/isaac_utils"
 cd "$ROOT/third_party/DreamControl/Training"
 extra=(); [[ "$STAGE" == witness ]] && extra+=(--witness)
 timeout --signal=TERM --kill-after=30s 18000s "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u "$ROOT/scripts/r3_feedback_paused_isaac_20261007.py" --protocol "$ROOT/configs/feedback_paused_isaac_20261007.json" --output "$OUT" --seed "$SEED" --device cuda:0 --headless "${extra[@]}"
 ;;
prepare|generate)
 export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$TRAJ"
 cd "$TRAJ"
 if [[ "$STAGE" == prepare ]]; then
  "$OLD/.venv-pbhc-training/bin/python" -u "$ROOT/scripts/r3_feedback_paused_generation_20261007.py" --stage prepare --output "$OUT"
 else
  case "$ARM" in planned|actual|shuffled) ;; *) exit 2 ;; esac
  timeout --signal=TERM --kill-after=30s 1800s "$OLD/.venv-pbhc-training/bin/python" -u "$ROOT/scripts/r3_feedback_paused_generation_20261007.py" --stage generate --arm "$ARM" --output "$OUT" --seed "$SEED" --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task kick --output_dir "$OUT/generation/$ARM" --batch_size 22
 fi
 ;;
retarget)
 case "$ARM" in planned|actual|shuffled) ;; *) exit 2 ;; esac
 TARGET="$TRAJ/sample_feedback_paused_20261007_seed${SEED}_$ARM"
 [[ ! -e "$TARGET" && -s "$OUT/generation/$ARM/results.npy" ]] || exit 2
 mkdir -p "$TARGET/Kick_sim" "$TARGET/Kick_sim1" "$TARGET/Kick_sim2"
 cp "$TRAJ/sample/Kick_sim/retarget.py" "$TARGET/Kick_sim/retarget.py"
 cp "$TRAJ/sample/Kick_sim1/refine_motions.py" "$TARGET/Kick_sim1/refine_motions.py"
 cp -a "$TRAJ/sample/retarget_helpers" "$TARGET/retarget_helpers"
 ln -s "$OUT/generation/$ARM/results.npy" "$TARGET/Kick_sim/results.npy"
 export XLA_PYTHON_CLIENT_PREALLOCATE=false
 export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
 export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$TRAJ/../Training/isaac_utils:$TARGET"
 (cd "$TARGET/Kick_sim" && timeout --signal=TERM --kill-after=30s 3600s "$OLD/.venv-retarget/bin/python" -u retarget.py)
 # Zero accepted inputs remain explicit failures. Do not manufacture references.
 count="$(find "$TARGET/Kick_sim1" -maxdepth 1 -type f -name '*.pkl' | wc -l)"
 if (( count > 0 )); then (cd "$TARGET/Kick_sim1" && timeout --signal=TERM --kill-after=30s 600s "$OLD/.venv-retarget/bin/python" -u refine_motions.py); fi
 echo "FEEDBACK_PAUSED_RETARGET_COMPLETE seed=$SEED arm=$ARM accepted=$count attempted=22"
 ;;
*) exit 2 ;;
esac
