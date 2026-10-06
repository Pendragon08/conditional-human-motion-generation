#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
MODE="${1:?witness or full}"; TAG="${2:?unique output tag}"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
if [[ "$MODE" == witness || "$MODE" == full ]]; then
  echo 'Native-camera route disabled after three exit139 witnesses. Preserve failures; use verified pose export only.'
  exit 2
fi
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd:$RUNTIME/third_party/sim-web-visualizer"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
export R3_CLOSD_SCENE_UDP=127.0.0.1:18446
MON="$ROOT/artifacts/monitoring"
exec 9>"$MON/r3_closd_reproduction_20261001.lock"
flock -n 9 || { echo 'Another CLoSD job holds the lock'; exit 1; }
STATE="$MON/r3_closd_record_${TAG}.state"
[[ ! -e "$STATE" ]] || { echo 'Attempt already exists; audit it, do not duplicate'; exit 1; }
trap 'rc=$?; if ((rc)); then echo "FAILED exit=$rc preserve_outputs=true" > "$STATE"; fi' EXIT
test "$(git -C "$ROOT/third_party/CLoSD" rev-parse HEAD)" = d8950bdb78312fcd2844f336b3fc079e7368c87f
"$RUNTIME/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_closd_env_witness.py"
ENTRY="$ROOT/scripts/r3_closd_record_frozen.py"
if [[ "$MODE" == pose_witness ]]; then
  FRAMES=60; MODELS=(CLoSD_multitask_finetune); ENTRY="$ROOT/scripts/r3_closd_record_poses.py"
elif [[ "$MODE" == poses ]]; then
  FRAMES=900; MODELS=(CLoSD_multitask_finetune CLoSD_no_finetune); ENTRY="$ROOT/scripts/r3_closd_record_poses.py"
elif [[ "$MODE" == witness ]]; then
  FRAMES=30; MODELS=(CLoSD_multitask_finetune)
elif [[ "$MODE" == full ]]; then
  FRAMES=900; MODELS=(CLoSD_multitask_finetune CLoSD_no_finetune)
else exit 2; fi
for model in "${MODELS[@]}"; do
  echo "RUNNING model=$model frames=$FRAMES inference_only=true" > "$STATE"
  OUT="$ROOT/artifacts/closd_recordings_20261006/$TAG/$model"
  timeout --signal=TERM --kill-after=45s 3600s "$RUNTIME/.venv-pbhc-training/bin/python" -u \
    "$ENTRY" --output "$OUT" --frames "$FRAMES" \
    learning=im_big env=closd_multitask robot=smpl_humanoid exp_name="$model" epoch=-1 \
    test=True env.episode_length=500 env.dip.cfg_param=7.5 env.num_envs=12 \
    headless=True no_virtual_display=True closd_eval=False \
    learning.params.config.player.games_num=100000 no_log=True debug=False seed=42 \
    > "$MON/r3_closd_record_${TAG}_${model}.log" 2>&1
  grep -q CLOSD_RECORDING_COMPLETE "$MON/r3_closd_record_${TAG}_${model}.log"
  python3 - "$OUT/manifest.json" <<'PY'
import json,sys
p=sys.argv[1];d=json.load(open(p));d['normal_exit']=True;d['status']='COMPLETE'
d['exit_code']=0;d['exit_verified_by']='parent_shell_after_process_exit'
with open(p,'w') as f:json.dump(d,f,indent=2)
PY
done
echo "COMPLETE exit=0 mode=$MODE inference_only=true" > "$STATE"
