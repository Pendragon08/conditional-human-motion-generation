#!/usr/bin/env bash
set -uo pipefail

# Official CLoSD snapshot, staged under /data after the user confirmed the
# separate SMPL, AMASS, and HumanML3D licences on 2026-09-26.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_closd_asset_download.state"
LOG="$MON/r3_closd_asset_download.log"
MANIFEST="$ROOT/artifacts/closd-r3/official_assets_manifest.json"
LOCK="$MON/r3_closd_asset_download.lock"
mkdir -p "$MON" "$ROOT/artifacts/closd-r3" "$ROOT/cache/closd_hf"
exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ -s "$MANIFEST" ]]; then
  printf 'COMPLETE manifest=%s\n' "$MANIFEST" > "$STATE"
  exit 0
fi

export HF_HOME="$ROOT/cache/closd_hf"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HTTPS_PROXY=http://127.0.0.1:18181
export HTTP_PROXY=http://127.0.0.1:18181
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export PYTHONPATH="$ROOT/envs/closd-r3/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export CLOSD_ASSET_MANIFEST="$MANIFEST"

printf 'RUNNING started=%s repo=guytevet/CLoSD\n' "$(date -u +%FT%TZ)" > "$STATE"
cd "$ROOT"
"$ROOT/.venv-pbhc-training/bin/python" -u scripts/r3_closd_fetch_assets.py > "$LOG" 2>&1
result=$?
if (( result != 0 )) || [[ ! -s "$MANIFEST" ]]; then
  printf 'BLOCKED download_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'COMPLETE manifest=%s\n' "$MANIFEST" > "$STATE"
