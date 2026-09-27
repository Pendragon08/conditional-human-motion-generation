#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTH="$ROOT/artifacts/closd-r3/xvfb-96.xauth"
mkdir -p "$(dirname "$AUTH")"
touch "$AUTH"
chmod 600 "$AUTH"
if ! xauth -f "$AUTH" list :96 | grep -q .; then
  xauth -f "$AUTH" add :96 . "$(openssl rand -hex 16)"
fi
exec "$ROOT/envs/closd-r3/xvfb/usr/bin/Xvfb" :96 \
  -screen 0 1280x720x24 -auth "$AUTH" -nolisten tcp -noreset
