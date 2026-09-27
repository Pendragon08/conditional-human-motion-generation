#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-10}"
SUFFIX="n100"
QUEUE_SUFFIX=""
if [[ "$SEED" != 10 ]]; then SUFFIX="${SUFFIX}_seed${SEED}"; QUEUE_SUFFIX="_seed${SEED}"; fi
MON="$ROOT/artifacts/monitoring"
UP="$MON/r3_dreamcontrol_retarget_${SUFFIX}.state"
STATE="$MON/r3_dreamcontrol_post_retarget_queue${QUEUE_SUFFIX}.state"
VALID="$MON/r3_dreamcontrol_stage_${SUFFIX}_validation.json"
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_r3_${SUFFIX}"
exec 9>"$MON/r3_dreamcontrol_post_retarget_queue_${SUFFIX}.lock"
flock -n 9 || exit 0
printf 'WAITING_FOR_RETARGET started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 270); do
  if grep -q '^FINISHED_NEEDS_VALIDATION' "$UP" 2>/dev/null; then
    break
  fi
  if grep -q '^BLOCKED' "$UP" 2>/dev/null; then
    printf 'BLOCKED retarget_failed upstream=%s\n' "$UP" > "$STATE"
    exit 1
  fi
  sleep 15
done
if ! grep -q '^FINISHED_NEEDS_VALIDATION' "$UP" 2>/dev/null; then
  printf 'BLOCKED retarget_wait_timeout upstream=%s\n' "$UP" > "$STATE"
  exit 1
fi
export PYTHONPATH="$ROOT/envs/dreamcontrol-retarget:$ROOT/third_party/pyroki/src:$ROOT/third_party/jaxls/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv-retarget/bin/python" "$ROOT/scripts/r3_dreamcontrol_validate_stage.py" --stage "$STAGE" --output "$VALID" > "$MON/r3_dreamcontrol_stage_${SUFFIX}_validation.log" 2>&1 || {
  printf 'BLOCKED retarget_validation_exception\n' > "$STATE"; exit 1;
}
"$ROOT/.venv-retarget/bin/python" - "$VALID" <<'PY' || { printf 'BLOCKED retarget_validation_failed\n' > "$STATE"; exit 1; }
import json, sys
d=json.load(open(sys.argv[1]))['sources']['Kick_sim1']
assert d['count']>0 and d['all_finite'] and d['all_strike_indices_valid']
assert d['total_joint_limit_violations']==0
PY
printf 'COMPLETE_VALIDATED output_count=%s report=%s\n' "$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)" "$VALID" > "$UP"
printf 'RUNNING_REFINE started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
bash "$ROOT/scripts/r3_dreamcontrol_refine_safe.sh" 100 "$SEED" || {
  printf 'BLOCKED refine_failed\n' > "$STATE"; exit 1;
}
"$ROOT/.venv-retarget/bin/python" "$ROOT/scripts/r3_dreamcontrol_validate_stage.py" --stage "$STAGE" --output "$VALID" >> "$MON/r3_dreamcontrol_stage_${SUFFIX}_validation.log" 2>&1 || {
  printf 'BLOCKED refine_validation_exception\n' > "$STATE"; exit 1;
}
"$ROOT/.venv-retarget/bin/python" - "$VALID" <<'PY' || { printf 'BLOCKED refine_validation_failed\n' > "$STATE"; exit 1; }
import json, sys
d=json.load(open(sys.argv[1]))['sources']
a,b=d['Kick_sim1'],d['Kick_sim2']
assert a['count']>0 and a['count']==b['count']
assert b['all_finite'] and b['all_strike_indices_valid']
assert b['total_joint_limit_violations']==0
PY
printf 'COMPLETE_VALIDATED output_count=%s report=%s\n' "$(find "$STAGE/Kick_sim2" -maxdepth 1 -name '*.pkl' -type f | wc -l)" "$VALID" > "$MON/r3_dreamcontrol_refine_${SUFFIX}.state"
printf 'COMPLETE_VALIDATED output_count=%s report=%s\n' "$(find "$STAGE/Kick_sim2" -maxdepth 1 -name '*.pkl' -type f | wc -l)" "$VALID" > "$STATE"
