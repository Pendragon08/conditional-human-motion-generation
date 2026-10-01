#!/usr/bin/env bash
# Continue after the visibility gate without repeating completed diagnostics/evals.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME=/data/code/ouyanghao/oyh/humanoid-combat-motion
MON="$ROOT/artifacts/monitoring"
exec 9>"$MON/r3_closd_reproduction_20261001.lock"
flock -n 9 || { echo 'Existing reproduction queue is still running'; exit 1; }
for stage in capacity3072 CLoSD_multitask_finetune_eval4096 CLoSD_no_finetune_eval4096; do
  grep -q '^COMPLETE ' "$MON/r3_closd_${stage}_20261001.state"
done
python3 - "$MON/r3_closd_3072_local_view_gate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('user_confirmed_local_stream') is True
assert d.get('source') == 'actual_isaacgym_simulation'
PY
export PYTHONPATH="$RUNTIME/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD:$ROOT/third_party/CLoSD/closd:$RUNTIME/third_party/sim-web-visualizer"
export PATH="$RUNTIME/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/closd-r3-torch-extensions"
export HF_HOME="$RUNTIME/cache/closd_hf" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONFAULTHANDLER=1
unset R3_CLOSD_WEB_FULL_SCENE R3_CLOSD_TELEMETRY_UDP
export R3_CLOSD_SCENE_UDP=127.0.0.1:18446
cd "$ROOT/third_party/CLoSD"
STATE="$MON/r3_closd_reproduction_20261001.state"
trap 'rc=$?; if ((rc)); then echo "BLOCKED exit=$rc preserve_logs=true" > "$STATE"; fi' EXIT
run_stage(){
  local name="$1"; shift
  local log="$MON/r3_closd_${name}_20261001.log"
  [[ ! -e "$log" ]] || { echo "Preexisting $name attempt requires audit" >&2; return 1; }
  echo "RUNNING stage=$name gpu=0" > "$STATE"
  timeout --signal=TERM --kill-after=45s 172800s "$RUNTIME/.venv-pbhc-training/bin/python" -u \
    "$ROOT/scripts/r3_closd_isolated_entry.py" "$@" > "$log" 2>&1
  grep -q CLOSD_NATIVE_CLEANUP_COMPLETE "$log"
  echo "COMPLETE stage=$name exit=0" > "$MON/r3_closd_${name}_20261001.state"
}
run_stage formal3072 learning=im_big env=closd_multitask robot=smpl_humanoid \
  learning.params.load_checkpoint=True learning.params.load_path=output/CLoSD/CLoSD_no_finetune/Humanoid.pth \
  env.dip.cfg_param=2.5 env.num_envs=3072 learning.params.config.max_epochs=66000 \
  has_eval=False epoch=0 headless=True no_virtual_display=True no_log=True debug=False seed=0 \
  exp_name=r3_CLoSD_official3072_20261001
run_stage retrained_eval4096 learning=im_big env=closd_multitask robot=smpl_humanoid \
  exp_name=r3_CLoSD_official3072_20261001 epoch=66000 test=True env.episode_length=500 \
  env.dip.cfg_param=7.5 env.num_envs=4096 no_virtual_display=True headless=True closd_eval=True \
  learning.params.config.player.games_num=100000 no_log=True debug=False seed=0
grep -q 'Collected enough data (1000 per task) success rates:' "$MON/r3_closd_retrained_eval4096_20261001.log"
grep -q 'Eval plot ended' "$MON/r3_closd_retrained_eval4096_20261001.log"
echo 'COMPLETE official3072_training_and_eval4096' > "$STATE"
