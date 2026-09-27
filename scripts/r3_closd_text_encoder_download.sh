#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_text_encoder_download.state"
LOG="$MON/r3_closd_text_encoder_download.log"
MANIFEST="$ROOT/artifacts/closd-r3/text_encoder_manifest.json"
LOCK="$MON/r3_closd_text_encoder_download.lock"
mkdir -p "$MON"
exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ -s "$MANIFEST" ]]; then
  printf 'COMPLETE manifest=%s\n' "$MANIFEST" > "$STATE"
  exit 0
fi
export PYTHONPATH="$ROOT/envs/closd-r3/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export HTTPS_PROXY=http://127.0.0.1:18181 HTTP_PROXY=http://127.0.0.1:18181
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export CLOSD_TEXT_ENCODER_MANIFEST="$MANIFEST"
printf 'RUNNING started=%s repo=distilbert/distilbert-base-uncased\n' "$(date -u +%FT%TZ)" > "$STATE"
cd "$ROOT"
"$ROOT/.venv-pbhc-training/bin/python" -u scripts/r3_closd_fetch_text_encoder.py > "$LOG" 2>&1
result=$?
if (( result != 0 )) || [[ ! -s "$MANIFEST" ]]; then
  printf 'BLOCKED download_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'COMPLETE manifest=%s\n' "$MANIFEST" > "$STATE"
