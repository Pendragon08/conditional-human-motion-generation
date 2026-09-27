#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
exec 9>"$MON/r3_dreamcontrol_augmented_resume100_queue.lock"
flock -n 9 || { echo 'Resume queue already running' >&2; exit 1; }
"$ROOT/scripts/r3_dreamcontrol_augmented_resume100.sh"
R3_AUG_TRAIN_STATE="$MON/r3_dreamcontrol_augmented_seed42_resume100_1900.state" \
R3_AUG_TRAIN_LOG="$MON/r3_dreamcontrol_augmented_seed42_resume100_1900.log" \
  "$ROOT/scripts/r3_dreamcontrol_augmented_posttrain_eval.sh"
