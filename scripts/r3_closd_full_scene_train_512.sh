#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_full_scene_train_512.state"
LOG="$MON/r3_closd_full_scene_train_512.log"
GATE="$MON/r3_closd_full_scene_gate.json"
CKPT="$ROOT/third_party/CLoSD/output/CLoSD/CLoSD_no_finetune/Humanoid.pth"
[[ -s "$CKPT" ]] || { echo 'BLOCKED pretrained_controller_missing' > "$STATE"; exit 1; }
[[ -s "$ROOT/artifacts/closd-r3/official_assets_manifest.json" ]] || {
  echo 'BLOCKED official_assets_missing' > "$STATE"; exit 1;
}
BASE_EPOCH="$("$ROOT/.venv-pbhc-training/bin/python" - "$CKPT" <<'PY'
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu")["epoch"]))
PY
)"
TARGET_EPOCH="$((BASE_EPOCH + 4000))"
python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get("status") == "passed"
assert gate.get("user_confirmed") is True
assert gate.get("source") == "isaacgym_full_scene_meshcat"
PY
[[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7000/static/)" == 200 ]] || {
  echo 'BLOCKED full_scene_server_unavailable' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/sim-web-visualizer:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1
export R3_CLOSD_WEB_FULL_SCENE=1 R3_CLOSD_WEB_MAX_ENVS=2 R3_CLOSD_WEB_RENDER_EVERY=4
export R3_CLOSD_TELEMETRY_UDP=127.0.0.1:18445
cd "$ROOT/third_party/CLoSD"
echo "RUNNING formal_pilot_finetune envs=512 base_epoch=$BASE_EPOCH target_epoch=$TARGET_EPOCH additional_epochs=4000 gpu_physical=1 full_isaac_scene=true monitor_separate=true source=official_no_finetune" > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 172800s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big env=closd_multitask robot=smpl_humanoid \
  learning.params.load_checkpoint=True \
  learning.params.load_path=output/CLoSD/CLoSD_no_finetune/Humanoid.pth \
  env.dip.cfg_param=2.5 env.num_envs=512 \
  learning.params.config.max_epochs="$TARGET_EPOCH" \
  has_eval=False epoch=0 headless=False no_virtual_display=True \
  no_log=True debug=False seed=0 \
  exp_name=r3_CLoSD_multitask_finetune_512_full_scene_v2 \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  printf 'FAILED train_exit=%s log=%s\n' "$rc" "$LOG" > "$STATE"
  exit "$rc"
fi
echo 'COMPLETE formal_pilot_finetune envs=512' > "$STATE"
