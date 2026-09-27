#!/usr/bin/env bash
set -euo pipefail

# Continue the interrupted augmented run without a renderer or WebRTC in the
# training process. A separate checkpoint viewer may fail independently.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
SOURCE_RUN="2026-09-27_01-09-52-Kick-a"
CKPT="$TRAIN/logs/rsl_rl/g1/$SOURCE_RUN/model_200.pt"
SNAPSHOT="$MON/r3_aug_seed42_resume_model200_snapshot.pt"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/augmented_train"
LOG="$MON/r3_dreamcontrol_augmented_seed42_resume200_1800.log"
STATE="$MON/r3_dreamcontrol_augmented_seed42_resume200_1800.state"
GATE="$MON/r3_seed42_local_view_gate.json"

python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get("status") == "passed" and gate.get("user_confirmed") is True
assert gate.get("dataset") == "augmented_train_seed42"
PY
[[ -s "$CKPT" && -s "$SNAPSHOT" ]] || { echo 'BLOCKED checkpoint_missing' > "$STATE"; exit 1; }
[[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" == "$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)" ]] || { echo 'BLOCKED checkpoint_hash_mismatch' > "$STATE"; exit 1; }
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 98 ]] || { echo 'BLOCKED reference_count' > "$STATE"; exit 1; }
if pgrep -af 'r3_dreamcontrol_train_livestream.py.*Isaac-Motion-Tracking-Kick-v0' | grep -v grep >/dev/null; then
  echo 'BLOCKED other_kick_trainer_running' > "$STATE"
  exit 1
fi

export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES PUBLIC_IP
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
echo "RUNNING source=$CKPT refs=$REF envs=1024 additional_iterations=1800 target=1999 seed=123 headless=true livestream=false" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 172800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --seed=123 --num_envs=1024 \
  --resume --load_run="$SOURCE_RUN" --checkpoint=model_200.pt \
  --max_iterations=1800 --headless \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  echo "BLOCKED train_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE iteration=1999/2000 source=$CKPT log=$LOG" > "$STATE"
