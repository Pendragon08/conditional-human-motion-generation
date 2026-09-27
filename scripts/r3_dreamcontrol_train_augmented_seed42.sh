#!/usr/bin/env bash
set -euo pipefail

# Fixed-budget 66+32 reference comparison against the completed base66 run.
# Start only after the local live-view gate has been recorded.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/augmented_train"
MON="$ROOT/artifacts/monitoring"
GATE="$MON/r3_seed42_local_view_gate.json"
LOG="$MON/r3_dreamcontrol_augmented_seed42_2000_1024.log"
STATE="$MON/r3_dreamcontrol_augmented_seed42_2000_1024.state"
mkdir -p "$MON"

if [[ ! -f "$GATE" ]]; then
  echo "BLOCKED local_live_view_not_confirmed" > "$STATE"
  exit 1
fi
python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get("status") == "passed" and gate.get("user_confirmed") is True
assert gate.get("dataset") == "augmented_train_seed42"
PY
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 98 ]] || { echo "BLOCKED reference_count" > "$STATE"; exit 1; }
if [[ -f "$STATE" ]] && grep -Eq '^(RUNNING|COMPLETE) ' "$STATE"; then
  echo "Existing train state: $(cat "$STATE")" >&2
  exit 1
fi

export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
echo "RUNNING refs=$REF envs=1024 iterations=2000 seed=123 gpu=cuda:0 livestream=2" > "$STATE"
cd "$TRAIN"
set +e
"$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --seed=123 --num_envs=1024 --max_iterations=2000 \
  --livestream=2 --video --video_length=200 --video_interval=2000 \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  echo "BLOCKED train_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE iteration=1999/2000 log=$LOG" > "$STATE"
