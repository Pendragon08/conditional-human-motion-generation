#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
UP="$MON/r3_dreamcontrol_seed42_queue.state"
STATE="$MON/r3_dreamcontrol_seed42_postquality.state"
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_r3_n100_seed42"
OLD="$ROOT/third_party/DreamControl/TrajGen/sample/Kick_sim2"
exec 9>"$MON/r3_dreamcontrol_seed42_postquality.lock"
flock -n 9 || exit 0
printf 'WAITING_FOR_REFINEMENT started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 300); do
  if grep -q '^COMPLETE_VALIDATED' "$UP" 2>/dev/null; then break; fi
  if grep -q '^BLOCKED' "$UP" 2>/dev/null; then
    printf 'BLOCKED upstream_failed\n' > "$STATE"; exit 1
  fi
  sleep 15
done
if ! grep -q '^COMPLETE_VALIDATED' "$UP" 2>/dev/null; then
  printf 'BLOCKED wait_timeout\n' > "$STATE"; exit 1
fi
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$ROOT/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/envs/dreamcontrol-retarget:$ROOT/third_party/pyroki/src:$ROOT/third_party/jaxls/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'RUNNING started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
"$ROOT/.venv-retarget/bin/python" "$ROOT/scripts/r3_dreamcontrol_crossset_novelty.py" \
  --supplied "$OLD" --candidate "$STAGE/Kick_sim2" \
  --output "$MON/r3_dreamcontrol_seed42_crossset_novelty.json" \
  > "$MON/r3_dreamcontrol_seed42_postquality.log" 2>&1 || {
  printf 'BLOCKED crossset_novelty_failed\n' > "$STATE"; exit 1;
}
"$ROOT/.venv-retarget/bin/python" "$ROOT/scripts/r3_dreamcontrol_contact_geometry.py" \
  --stage "$STAGE" --output "$MON/r3_dreamcontrol_seed42_contact_geometry.json" \
  >> "$MON/r3_dreamcontrol_seed42_postquality.log" 2>&1 || {
  printf 'BLOCKED contact_geometry_failed\n' > "$STATE"; exit 1;
}
printf 'COMPLETE_VALIDATED novelty=%s geometry=%s\n' "$MON/r3_dreamcontrol_seed42_crossset_novelty.json" "$MON/r3_dreamcontrol_seed42_contact_geometry.json" > "$STATE"
