#!/usr/bin/env bash
set -euo pipefail

# Resume the seed42-augmented G1 run from its verified model_100.pt.
# 1900 additional iterations reach global iteration 1999/2000.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
OLD_RUN="2026-09-26_16-11-45-Kick-a"
CKPT="$TRAIN/logs/rsl_rl/g1/$OLD_RUN/model_100.pt"
SNAPSHOT="$MON/r3_aug_seed42_stall_model100_snapshot.pt"
REF="$ROOT/third_party/DreamControl/TrajGen/r3_comparison_seed42/augmented_train"
GATE="$MON/r3_seed42_local_view_gate.json"
LOG="$MON/r3_dreamcontrol_augmented_seed42_resume100_1900.log"
STATE="$MON/r3_dreamcontrol_augmented_seed42_resume100_1900.state"

if [[ -f "$STATE" ]] && grep -Eq '^(RUNNING|COMPLETE) ' "$STATE"; then
  echo "Existing recovery state: $(cat "$STATE")" >&2
  exit 1
fi
python3 - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get("status") == "passed" and gate.get("user_confirmed") is True
assert gate.get("dataset") == "augmented_train_seed42"
PY
[[ -s "$CKPT" && -s "$SNAPSHOT" ]] || { echo 'BLOCKED checkpoint_missing' > "$STATE"; exit 1; }
[[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" == "$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)" ]] || { echo 'BLOCKED checkpoint_hash_mismatch' > "$STATE"; exit 1; }
[[ "$(find "$REF" -maxdepth 1 -name '*.pkl' | wc -l)" -eq 98 ]] || { echo 'BLOCKED reference_count' > "$STATE"; exit 1; }

export DREAMCONTROL_KICK_REF_PATH="$REF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" PIP_CACHE_DIR="$ROOT/cache/r3_pip"
export XDG_CACHE_HOME="$ROOT/cache/r3_xdg" MPLCONFIGDIR="$ROOT/cache/r3_matplotlib"
export PUBLIC_IP=10.147.17.32 WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
python3 "$ROOT/scripts/build_dreamcontrol_livestream_entry.py"
echo "RUNNING source=$CKPT refs=$REF envs=1024 additional_iterations=1900 target=1999 seed=123 livestream=2" > "$STATE"
cd "$TRAIN"
set +e
timeout --signal=TERM --kill-after=45s 172800s \
  "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
  "$ROOT/scripts/r3_dreamcontrol_train_livestream.py" \
  --task=Isaac-Motion-Tracking-Kick-v0 \
  --device=cuda:0 --seed=123 --num_envs=1024 \
  --resume --load_run="$OLD_RUN" --checkpoint=model_100.pt \
  --max_iterations=1900 --livestream=2 \
  --video --video_length=200 --video_interval=2000 \
  > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
  echo "BLOCKED train_exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE iteration=1999/2000 source=$CKPT log=$LOG" > "$STATE"
