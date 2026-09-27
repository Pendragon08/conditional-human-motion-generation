#!/usr/bin/env bash
set -uo pipefail

# Download only the author's linked OmniControl/DreamControl generator checkpoint.
# Keep the original filename in a private /data staging directory for inspection.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/dreamcontrol-r3/download"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_generator_checkpoint.state"
LOG="$MON/r3_dreamcontrol_generator_checkpoint.log"
LOCK="$MON/r3_dreamcontrol_generator_checkpoint.lock"
mkdir -p "$OUT" "$MON"
exec 9>"$LOCK"
flock -n 9 || exit 0
if find "$OUT" -maxdepth 1 -type f -size +1M | grep -q .; then
  printf 'NEEDS_INSPECTION existing_file_in=%s\n' "$OUT" > "$STATE"
  exit 0
fi
export PYTHONPATH="$ROOT/envs/dreamcontrol-gdown${PYTHONPATH:+:$PYTHONPATH}"
export HTTPS_PROXY=http://127.0.0.1:18181 HTTP_PROXY=http://127.0.0.1:18181
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
printf 'RUNNING started=%s source=official_DreamControl_TrajGen_README\n' "$(date -u +%FT%TZ)" > "$STATE"
"$ROOT/.venv-pbhc-training/bin/python" -m gdown \
  --id 1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud --output "$OUT/" \
  > "$LOG" 2>&1
result=$?
if (( result != 0 )); then
  printf 'BLOCKED download_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'NEEDS_INSPECTION completed=%s output=%s\n' "$(date -u +%FT%TZ)" "$OUT" > "$STATE"
find "$OUT" -maxdepth 1 -type f -printf '%f %s bytes\n' >> "$LOG"
