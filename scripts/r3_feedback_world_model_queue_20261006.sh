#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MON="$ROOT/artifacts/monitoring"
mkdir -p "$MON"
exec 8>"$MON/r3_feedback_world_model_queue_20261006.lock"
flock -n 8 || { echo 'Queue is already active' >&2; exit 1; }
STATE="$MON/r3_feedback_world_model_queue_20261006.state"
[[ ! -e "$ROOT/results/feedback_world_model_20261006/collection.json" ]] || { echo 'Existing collection must be preserved' >&2; exit 1; }
for stage in collect fit; do
  echo "RUNNING stage=$stage utc=$(date -u +%FT%TZ) no_ppo=true" > "$STATE"
  set +e
  timeout --signal=TERM --kill-after=30s 3600s bash scripts/r3_feedback_world_model_pilot.sh "$stage" > "$MON/r3_feedback_world_model_${stage}_20261006.log" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "FAILED stage=$stage exit=$rc utc=$(date -u +%FT%TZ)" > "$STATE"
    exit "$rc"
  fi
done
echo "COMPLETE stages=collect,fit utc=$(date -u +%FT%TZ) no_control_benefit_claim=true" > "$STATE"
