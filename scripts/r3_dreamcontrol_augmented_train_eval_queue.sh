#!/usr/bin/env bash
set -euo pipefail

# One queue for a single formal training run followed by its fixed held-out eval.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
mkdir -p "$MON"
exec 9>"$MON/r3_dreamcontrol_augmented_seed42_queue.lock"
flock -n 9 || { echo 'Queue already running' >&2; exit 1; }
"$ROOT/scripts/r3_dreamcontrol_train_augmented_seed42.sh"
"$ROOT/scripts/r3_dreamcontrol_augmented_posttrain_eval.sh"
