#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_seed42_queue.state"
UP="$MON/r3_dreamcontrol_generate_n100_seed42.state"
BASE="$ROOT/artifacts/dreamcontrol-r3/generated/kick_guided_n100_seed10/results.npy"
NEW="$ROOT/artifacts/dreamcontrol-r3/generated/kick_guided_n100_seed42/results.npy"
NOVELTY="$MON/r3_dreamcontrol_seed42_human_novelty.json"
exec 9>"$MON/r3_dreamcontrol_seed42_queue.lock"
flock -n 9 || exit 0
printf 'WAITING_FOR_GENERATION started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 150); do
  if grep -q '^FINISHED_NEEDS_VALIDATION' "$UP" 2>/dev/null; then break; fi
  if grep -q '^BLOCKED' "$UP" 2>/dev/null; then
    printf 'BLOCKED generation_failed\n' > "$STATE"; exit 1
  fi
  sleep 15
done
if ! grep -q '^FINISHED_NEEDS_VALIDATION' "$UP" 2>/dev/null; then
  printf 'BLOCKED generation_wait_timeout\n' > "$STATE"; exit 1
fi
"$ROOT/.venv-pbhc-training/bin/python" "$ROOT/scripts/r3_dreamcontrol_human_novelty.py" \
  --baseline "$BASE" --candidate "$NEW" --output "$NOVELTY" || {
  printf 'BLOCKED generation_validation_or_novelty_failed\n' > "$STATE"; exit 1;
}
printf 'COMPLETE_VALIDATED count=100 seed=42 novelty_report=%s\n' "$NOVELTY" > "$UP"
bash "$ROOT/scripts/r3_dreamcontrol_stage_retarget.sh" 100 42 || {
  printf 'BLOCKED stage_failed\n' > "$STATE"; exit 1;
}
printf 'RUNNING_RETARGET started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
bash "$ROOT/scripts/r3_dreamcontrol_retarget_safe.sh" 100 42 || {
  printf 'BLOCKED retarget_failed\n' > "$STATE"; exit 1;
}
bash "$ROOT/scripts/r3_dreamcontrol_post_retarget_queue.sh" 42 || {
  printf 'BLOCKED post_retarget_failed\n' > "$STATE"; exit 1;
}
printf 'COMPLETE_VALIDATED output=%s\n' "$ROOT/third_party/DreamControl/TrajGen/sample_r3_n100_seed42/Kick_sim2" > "$STATE"
