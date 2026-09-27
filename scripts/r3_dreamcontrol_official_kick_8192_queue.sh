#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_official_kick_8192_queue.state"
mkdir -p "$MON"
exec 9>"$MON/r3_dreamcontrol_official_kick_8192_queue.lock"
flock -n 9 || { echo 'Official Kick queue already running' >&2; exit 1; }

# Preserve the already-running augmented comparison and its held-out evaluation.
echo 'QUEUED waiting_for_existing_augmented_train_and_eval' > "$STATE"
deadline=$((SECONDS + 43200))
while tmux has-session -t r3_dreamcontrol_aug_seed42_resume200 2>/dev/null; do
  (( SECONDS < deadline )) || { echo 'BLOCKED existing_queue_timeout' > "$STATE"; exit 1; }
  sleep 30
done

# Do not collide with any other GPU-0 job. The separate GPU-1 viewer may use
# a small amount of GPU-0 memory for rendering, so use a 1 GiB allowance.
deadline=$((SECONDS + 3600))
while :; do
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then break; fi
  (( SECONDS < deadline )) || { echo "BLOCKED gpu0_busy used_mib=$used" > "$STATE"; exit 1; }
  sleep 30
done

echo 'RUNNING official66_8192_2000' > "$STATE"
"$ROOT/scripts/r3_dreamcontrol_official_kick_8192.sh"
"$ROOT/scripts/r3_dreamcontrol_official_kick_8192_eval.sh"
echo 'COMPLETE official_training_and_evaluation' > "$STATE"
