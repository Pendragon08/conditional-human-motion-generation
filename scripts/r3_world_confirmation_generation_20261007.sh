#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD="$(dirname "$ROOT")/humanoid-combat-motion"
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
ART="$ROOT/artifacts/world_confirmation_20261007"
STAGE="$TRAJ/sample_world_confirm_20261007"
PREFIX="$ROOT/artifacts/monitoring/r3_world_confirmation_generation_20261007"
[[ ! -e "$ART" && ! -e "$STAGE" && ! -e "$PREFIX.state" ]] || { echo 'Preserve existing invocation'; exit 1; }
exec 9>"$ROOT/artifacts/monitoring/r3_world_ppo_gpu0.lock"
flock -n 9 || { echo 'GPU0_BUSY'; exit 1; }
used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
(( used < 1500 )) || { echo "BLOCKED_GPU0 used=$used"; exit 1; }
mkdir -p "$ART" "$STAGE/Kick_sim" "$STAGE/Kick_sim1" "$STAGE/Kick_sim2"
echo 'RUNNING stage=generate32 formal_PPO=false' > "$PREFIX.state"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT/scripts:$OLD/envs/closd-r3/site-packages:$OLD/third_party/CLIP:$TRAJ"
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
cd "$TRAJ"
if ! timeout --signal=TERM --kill-after=30s 300 "$OLD/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_world_confirmation_runtime_witness.py" --mode generation --output "$ART/generation_kernel.json" > "$PREFIX.generation_kernel.log" 2>&1; then
 echo 'FAILED stage=generation_kernel preserve=true' > "$PREFIX.state"; exit 1
fi
[[ "$(sha256sum save/omnicontrol_ckpt/model_humanml3d.pt | cut -d ' ' -f1)" == 886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0 ]] || { echo 'FAILED stage=checkpoint_hash preserve=true' > "$PREFIX.state"; exit 1; }
set +e
timeout --signal=TERM --kill-after=30s 1800 "$OLD/.venv-pbhc-training/bin/python" -u \
 "$ROOT/scripts/r3_dreamcontrol_generate_safe.py" --r3_limit 32 --seed 20261007 \
 --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 \
 --task kick --batch_size 128 --output_dir "$ART/generated" > "$PREFIX.generate.log" 2>&1
rc=$?
set -e
if (( rc )); then echo "FAILED stage=generate exit=$rc preserve=true" > "$PREFIX.state"; exit 1; fi
cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
cp "$TRAJ/sample/Kick_sim1/refine_motions.py" "$STAGE/Kick_sim1/refine_motions.py"
cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
ln -s "$ART/generated/results.npy" "$STAGE/Kick_sim/results.npy"
export LD_LIBRARY_PATH="$OLD/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$OLD/envs/dreamcontrol-retarget:$OLD/third_party/pyroki/src:$OLD/third_party/jaxls/src:$ROOT/third_party/DreamControl/Training/isaac_utils:$STAGE"
if ! timeout --signal=TERM --kill-after=30s 300 "$OLD/.venv-retarget/bin/python" "$ROOT/scripts/r3_world_confirmation_runtime_witness.py" --mode retarget --output "$ART/retarget_kernel.json" > "$PREFIX.retarget_kernel.log" 2>&1; then
 echo 'FAILED stage=retarget_kernel preserve=true' > "$PREFIX.state"; exit 1
fi
echo 'RUNNING stage=original_retarget formal_PPO=false' > "$PREFIX.state"
cd "$STAGE/Kick_sim"
set +e
timeout --signal=TERM --kill-after=30s 3600 "$OLD/.venv-retarget/bin/python" -u retarget.py > "$PREFIX.retarget.log" 2>&1
rc=$?
set -e
if (( rc )); then echo "FAILED stage=retarget exit=$rc preserve=true" > "$PREFIX.state"; exit 1; fi
echo 'RUNNING stage=original_refine formal_PPO=false' > "$PREFIX.state"
cd "$STAGE/Kick_sim1"
set +e
timeout --signal=TERM --kill-after=30s 900 "$OLD/.venv-retarget/bin/python" -u refine_motions.py > "$PREFIX.refine.log" 2>&1
rc=$?
set -e
if (( rc )); then echo "FAILED stage=refine exit=$rc preserve=true" > "$PREFIX.state"; exit 1; fi
echo 'RUNNING stage=static_selection formal_PPO=false' > "$PREFIX.state"
set +e
timeout --signal=TERM --kill-after=30s 600 "$OLD/.venv-retarget/bin/python" -u \
 "$ROOT/scripts/r3_world_confirmation_references_20261007.py" --stage "$STAGE" --output "$ART/selection" > "$PREFIX.selection.log" 2>&1
rc=$?
set -e
if (( rc )) || ! grep -q WORLD_FRESH_CONFIRMATION_REFERENCES_COMPLETE "$PREFIX.selection.log"; then
 echo "FAILED stage=static_selection exit=$rc preserve=true" > "$PREFIX.state"; exit 1
fi
echo 'COMPLETE stage=references exit=0 formal_PPO=false' > "$PREFIX.state"
echo 'WORLD_CONFIRMATION_GENERATION_COMPLETE'
