#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
LOG="${R3_AUG_TB_TRAIN_LOG:-$ROOT/artifacts/monitoring/r3_dreamcontrol_augmented_seed42_2000_1024.log}"
TB_LOG="${R3_AUG_TB_OUT_LOG:-$ROOT/artifacts/monitoring/r3_dreamcontrol_augmented_tensorboard.log}"
run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$LOG" | tail -n 1 | tr -d '\r')"
[[ -n "$run_name" && -d "$TRAIN/logs/rsl_rl/g1/$run_name" ]] || { echo 'Training run not found' >&2; exit 1; }
exec "$ROOT/.venv-pbhc-training/bin/tensorboard" \
  --logdir "$TRAIN/logs/rsl_rl/g1/$run_name" \
  --host 127.0.0.1 --port 6009 > "$TB_LOG" 2>&1
