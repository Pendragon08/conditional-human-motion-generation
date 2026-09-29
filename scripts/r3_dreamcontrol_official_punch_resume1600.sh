#!/usr/bin/env bash
set -euo pipefail

# Resume only the interrupted official Punch run. Training does not own the
# WebRTC stream; a separate checkpoint replay may be restarted at any time.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
SOURCE_RUN="2026-09-27_07-24-06-Punch-a"
CKPT="$TRAIN/logs/rsl_rl/g1/$SOURCE_RUN/model_1600.pt"
SNAPSHOT="$MON/r3_dreamcontrol_official_Punch_model1600_snapshot.pt"
LOG="$MON/r3_dreamcontrol_official_Punch_8192_resume1600_train.log"
STATE="$MON/r3_dreamcontrol_official_Punch_8192_resume1600.state"
EVAL_LOG="$MON/r3_dreamcontrol_official_Punch_8192_eval.log"
TASK_STATE="$MON/r3_dreamcontrol_official_Punch_8192.state"
GATE="$MON/r3_dreamcontrol_other_tasks_local_view_gate.json"
mkdir -p "$MON"
exec 9>"$MON/r3_dreamcontrol_official_Punch_resume1600.lock"
flock -n 9 || { echo 'Punch resume already running' >&2; exit 1; }
trap 'rc=$?; if (( rc != 0 )); then echo "BLOCKED resume_exit=$rc log=$LOG" > "$STATE"; fi' EXIT

python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate['status'] in ('passed', 'authorized_after_preflight')
assert gate['user_authorized_start'] is True
assert gate['view'] == 'actual_training_simulation'
assert gate['first_task'] == 'Punch'
PY
[[ -s "$CKPT" ]] || { echo 'BLOCKED checkpoint_missing' > "$STATE"; exit 1; }
if [[ ! -e "$SNAPSHOT" ]]; then cp -p "$CKPT" "$SNAPSHOT"; fi
[[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" == "$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)" ]] || {
  echo 'BLOCKED checkpoint_hash_mismatch' > "$STATE"; exit 1;
}
[[ "$(find "$ROOT/third_party/DreamControl/TrajGen/sample/Punch_sim2" -maxdepth 1 -name '*.pkl' -type f | wc -l)" -eq 100 ]] || {
  echo 'BLOCKED reference_count_mismatch' > "$STATE"; exit 1;
}
if pgrep -af 'r3_dreamcontrol_train_livestream.py.*Isaac-Motion-Tracking-Punch-v0' | grep -v grep >/dev/null; then
  echo 'BLOCKED other_punch_trainer_running' > "$STATE"; exit 1
fi

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
unset PUBLIC_IP
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"

echo "RUNNING source=$CKPT envs=8192 additional_iterations=400 target=1999 seed=42 gpu=0 headless=true" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 172800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Punch-v0 --device=cuda:0 \
  --num_envs=8192 --seed=42 \
  --resume --load_run="$SOURCE_RUN" --checkpoint=model_1600.pt \
  --max_iterations=400 --headless \
  > "$LOG" 2>&1
train_rc=$?
set -e
if (( train_rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  echo "BLOCKED train_exit=$train_rc log=$LOG" > "$STATE"; exit 1
fi
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$LOG" | tail -n 1 | tr -d '\r')"
final="$TRAIN/logs/rsl_rl/g1/$run_name/model_1999.pt"
[[ -n "$run_name" && -s "$final" ]] || { echo "BLOCKED final_checkpoint_missing run=$run_name" > "$STATE"; exit 1; }
echo "RUNNING stage=eval checkpoint=$final" > "$STATE"
set +e
timeout --signal=TERM --kill-after=45s 3600s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  scripts/reinforcement_learning/rsl_rl/play_eval.py \
  --task=Isaac-Motion-Tracking-Punch-v0 --device=cuda:0 \
  --num_envs=1000 --path="$final" --headless --video --video_length=500 \
  --id=r3_official_Punch_8192_eval > "$EVAL_LOG" 2>&1
eval_rc=$?
set -e
if (( eval_rc != 0 )) || ! grep -q 'Success rate:' "$EVAL_LOG"; then
  echo "BLOCKED eval_exit=$eval_rc log=$EVAL_LOG" > "$STATE"; exit 1
fi
video="$(find "$TRAIN/logs/rsl_rl/g1/$run_name/videos/play" -type f -name '*r3_official_Punch_8192_eval*.mp4' -size +0c | head -n 1)"
[[ -n "$video" ]] || { echo 'BLOCKED no_video' > "$STATE"; exit 1; }
success="$(grep 'Success rate:' "$EVAL_LOG" | tail -n 1)"
echo "COMPLETE task=Punch $success checkpoint=$final video=$video resumed_from=$CKPT" > "$TASK_STATE"
echo "COMPLETE iteration=1999/2000 task=Punch $success checkpoint=$final video=$video" > "$STATE"
