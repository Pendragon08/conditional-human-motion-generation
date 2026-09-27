#!/usr/bin/env bash
set -euo pipefail

# Sequential official DreamControl baselines for the ten non-Kick tasks.
# This queue is gated on a locally confirmed view of the actual training
# simulation. The existing CLoSD training on GPU 1 is never touched.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
GATE="$MON/r3_dreamcontrol_other_tasks_local_view_gate.json"
QUEUE_STATE="$MON/r3_dreamcontrol_official_other_tasks_queue.state"
mkdir -p "$MON"
exec 9>"$MON/r3_dreamcontrol_official_other_tasks_queue.lock"
flock -n 9 || { echo 'Official other-task queue already running' >&2; exit 1; }
trap 'rc=$?; if (( rc != 0 )); then echo "BLOCKED queue_exit=$rc see_task_state_files" > "$QUEUE_STATE"; fi' EXIT

python3 - "$GATE" <<'PY'
import json, pathlib, sys
g = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert g['status'] in ('passed', 'authorized_after_preflight')
assert g['user_authorized_start'] is True
assert g['view'] == 'actual_training_simulation'
assert g['first_task'] == 'Punch'
PY

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
unset DREAMCONTROL_KICK_REF_PATH
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"

tasks=(Punch Jump Sit Pick Bimanual-Pick Ground-Pick Ground-Pick-Top Button-Press Open-Drawer Open-Door)
ids=(
  Isaac-Motion-Tracking-Punch-v0
  Isaac-Motion-Tracking-Jump-v0
  Isaac-Motion-Tracking-Sit-v0
  Isaac-Motion-Tracking-Pick-v0
  Isaac-Motion-Tracking-Bimanual-Pick-v0
  Isaac-Motion-Tracking-Ground-Pick-v0
  Isaac-Motion-Tracking-Ground-Pick-Top-v0
  Isaac-Motion-Tracking-Button-Press-v0
  Isaac-Motion-Tracking-Open-Drawer-v0
  Isaac-Motion-Tracking-Open-Door-v0
)
refs=(Punch_sim2 Jump_sim2 Sit_sim2 Pick_sim2 Bimanual_Pick_sim2 Ground_Pick_sim2 Ground_Pick_Top_sim2 Button_Press_sim2 Open_Drawer_sim2 Open_Door_sim2)
counts=(100 1 1 67 1 1 1 96 602 602)

echo 'RUNNING task_count=10 envs=8192 iterations=2000 seed=42 gpu=0 livestream=2' > "$QUEUE_STATE"
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"; task_id="${ids[$i]}"; ref="${refs[$i]}"; count="${counts[$i]}"
  ref_dir="$ROOT/third_party/DreamControl/TrajGen/sample/$ref"
  state="$MON/r3_dreamcontrol_official_${task}_8192.state"
  train_log="$MON/r3_dreamcontrol_official_${task}_8192_train.log"
  eval_log="$MON/r3_dreamcontrol_official_${task}_8192_eval.log"
  if [[ -f "$state" ]] && grep -q '^COMPLETE ' "$state"; then continue; fi
  [[ "$(find "$ref_dir" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq "$count" ]] || {
    echo "BLOCKED task=$task reference_count_mismatch" > "$state"; exit 1;
  }
  # The renderer and task object assets are loaded by the official task; if
  # either fails, retain its diagnostic and stop the dependent queue.
  echo "RUNNING task=$task stage=train envs=8192 iterations=2000 seed=42" > "$state"
  cd "$TRAIN"
  set +e
  timeout --signal=TERM --kill-after=45s 172800s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
    --task="$task_id" --device=cuda:0 --num_envs=8192 \
    --max_iterations=2000 --seed=42 --livestream=2 \
    --kit_args="--portable-root $ROOT/cache/r3_official_kit --/renderer/multiGpu/enabled=false" \
    > "$train_log" 2>&1
  train_rc=$?
  set -e
  # Isaac Sim's python.sh wrapper can leave its Kit child running after a
  # timeout. Clean only this task's exact child, never CLoSD or other jobs.
  orphan_count="$(python3 - "$ROOT" "$task_id" <<'PY'
import os, signal, sys, time
root, task = sys.argv[1:]
target = (root + '/scripts/r3_dreamcontrol_train_livestream.py').encode()
task_arg = ('--task=' + task).encode()
found = []
for name in os.listdir('/proc'):
    if not name.isdigit() or int(name) == os.getpid():
        continue
    try:
        argv = open('/proc/' + name + '/cmdline', 'rb').read().split(b'\0')
    except (OSError, PermissionError):
        continue
    if target in argv and task_arg in argv:
        found.append(int(name))
for pid in found:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2 if found else 0)
for pid in found:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print(len(found))
PY
)"
  if (( orphan_count != 0 )); then
    echo "BLOCKED task=$task orphan_train_cleaned=$orphan_count train_exit=$train_rc log=$train_log" > "$state"; exit 1
  fi
  if (( train_rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$train_log"; then
    echo "BLOCKED task=$task train_exit=$train_rc log=$train_log" > "$state"; exit 1
  fi
  run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$train_log" | tail -n 1 | tr -d '\r')"
  ckpt="$TRAIN/logs/rsl_rl/g1/$run_name/model_1999.pt"
  [[ -n "$run_name" && -s "$ckpt" ]] || {
    echo "BLOCKED task=$task checkpoint_missing run=$run_name" > "$state"; exit 1;
  }
  echo "RUNNING task=$task stage=eval checkpoint=$ckpt" > "$state"
  # Official 1000-environment/500-frame evaluation with a non-empty video.
  set +e
  timeout --signal=TERM --kill-after=45s 3600s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    scripts/reinforcement_learning/rsl_rl/play_eval.py \
    --task="$task_id" --device=cuda:0 --num_envs=1000 \
    --path="$ckpt" --headless --video --video_length=500 \
    --id="r3_official_${task}_8192_eval" > "$eval_log" 2>&1
  eval_rc=$?
  set -e
  if (( eval_rc != 0 )) || ! grep -q 'Success rate:' "$eval_log"; then
    echo "BLOCKED task=$task eval_exit=$eval_rc log=$eval_log" > "$state"; exit 1
  fi
  video="$(find "$TRAIN/logs/rsl_rl/g1/$run_name/videos/play" -type f -name "*r3_official_${task}_8192_eval*.mp4" -size +0c | head -n 1)"
  [[ -n "$video" ]] || { echo "BLOCKED task=$task no_video" > "$state"; exit 1; }
  success="$(grep 'Success rate:' "$eval_log" | tail -n 1)"
  echo "COMPLETE task=$task $success checkpoint=$ckpt video=$video" > "$state"
  sleep 10
done
echo 'COMPLETE task_count=10 training_and_evaluation' > "$QUEUE_STATE"
