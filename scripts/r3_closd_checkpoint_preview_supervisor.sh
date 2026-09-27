#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_checkpoint_preview_supervisor.state"
mkdir -p "$MON"

# A preview started before this supervisor must finish before another is launched.
while tmux has-session -t r3_closd_checkpoint_preview 2>/dev/null; do
  echo "WAITING existing_checkpoint_preview training_unaffected=true" > "$STATE"
  sleep 10
done

failures=0
while true; do
  if grep -q '^COMPLETE formal_pilot_finetune' "$MON/r3_closd_full_scene_train_512.state" 2>/dev/null; then
    echo "STOPPED training_complete" > "$STATE"
    exit 0
  fi
  start="$(date +%s)"
  echo "RUNNING refreshing_checkpoint_preview training_unaffected=true" > "$STATE"
  if bash "$ROOT/scripts/r3_closd_checkpoint_preview.sh"; then
    failures=0
  else
    elapsed="$(( $(date +%s) - start ))"
    if (( elapsed < 120 )); then failures="$((failures + 1))"; else failures=0; fi
    if (( failures >= 3 )); then
      echo "BLOCKED repeated_preview_failures training_unaffected=true" > "$STATE"
      exit 1
    fi
  fi
  sleep 20
done
