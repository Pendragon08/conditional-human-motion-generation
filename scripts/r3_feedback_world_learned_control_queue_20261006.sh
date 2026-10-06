#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_feedback_world_learned_control_20261006.state"
[[ ! -e results/feedback_world_model_20261006/learned_control_summary.json && ! -e results/feedback_world_model_20261006/learned_control_records_partial.json ]] || exit 1
echo "RUNNING utc=$(date -u +%FT%TZ) no_ppo=true models_frozen=true" > "$STATE"
sha256sum scripts/r3_feedback_world_learned_control_20261006.py scripts/r3_feedback_world_oracle_20261006.py configs/feedback_world_learned_control_20261006.json configs/feedback_world_oracle_20261006.json > "$MON/r3_feedback_world_learned_control_20261006.source.sha256"
set +e
timeout --signal=TERM --kill-after=30s 3600s bash scripts/r3_feedback_world_learned_control_20261006.sh eval > "$MON/r3_feedback_world_learned_control_20261006.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  echo "COMPLETE utc=$(date -u +%FT%TZ) exit=0 full_innovation_validated=false" > "$STATE"
else
  echo "FAILED utc=$(date -u +%FT%TZ) exit=$rc" > "$STATE"
fi
exit "$rc"
