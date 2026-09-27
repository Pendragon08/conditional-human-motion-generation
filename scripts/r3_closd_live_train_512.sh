#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_live_training.state"
LOG="$MON/r3_closd_live_training.log"
GATE="$MON/r3_closd_local_view_gate.json"
CKPT="$ROOT/third_party/CLoSD/output/CLoSD/CLoSD_no_finetune/Humanoid.pth"
[[ -s "$CKPT" ]] || { echo 'BLOCKED pretrained_phc_missing' > "$STATE"; exit 1; }
python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get('status') == 'passed' and gate.get('user_confirmed') is True
assert gate.get('source') == 'closd_live_dashboard'
assert gate.get('alternative_monitor_accepted') is True, 'Browser skeleton monitor is not the native Isaac Gym GPU viewer'
PY
[[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6011/state)" == 200 ]] || {
  echo 'BLOCKED local_dashboard_unavailable' > "$STATE"; exit 1;
}
export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
export R3_CLOSD_SERIAL_MOTION_LOADING=1
export R3_CLOSD_TELEMETRY_UDP=127.0.0.1:18445
cd "$ROOT/third_party/CLoSD"
echo 'RUNNING pilot_finetune envs=512 target_epochs=4000 gpu_physical=1 monitor_separate=true source=official_no_finetune' > "$STATE"
set +e
timeout --signal=TERM --kill-after=30s 172800s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big env=closd_multitask robot=smpl_humanoid \
  learning.params.load_checkpoint=True \
  learning.params.load_path=output/CLoSD/CLoSD_no_finetune/Humanoid.pth \
  env.dip.cfg_param=2.5 env.num_envs=512 \
  learning.params.config.max_epochs=4000 \
  has_eval=False epoch=0 headless=True no_virtual_display=True \
  no_log=True debug=False seed=0 \
  exp_name=r3_CLoSD_multitask_finetune_512 \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  echo "BLOCKED train_exit=$rc log=$LOG" > "$STATE"; exit 1
fi
echo "COMPLETE pilot_finetune envs=512 log=$LOG" > "$STATE"
