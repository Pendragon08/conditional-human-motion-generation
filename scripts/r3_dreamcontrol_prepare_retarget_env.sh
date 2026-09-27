#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/envs/dreamcontrol-retarget"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_retarget_env.state"
LOG="$MON/r3_dreamcontrol_retarget_env.log"
mkdir -p "$OUT" "$MON"
if [[ -s "$OUT/jax/__init__.py" && -s "$OUT/viser/__init__.py" ]]; then
  printf 'NEEDS_IMPORT_VALIDATION target=%s\n' "$OUT" > "$STATE"
  exit 0
fi
printf 'RUNNING started=%s target=%s\n' "$(date -u +%FT%TZ)" "$OUT" > "$STATE"
"$ROOT/.venv-retarget/bin/python" -m pip install --target "$OUT" \
  'numpy==1.26.4' 'jax[cuda12]==0.6.2' jaxlie jax_dataclasses \
  jaxtyping loguru robot_descriptions yourdfpy 'viser==0.2.23' \
  pyliblzfse termcolor rich \
  > "$LOG" 2>&1
result=$?
if (( result != 0 )); then
  printf 'BLOCKED install_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'NEEDS_IMPORT_VALIDATION target=%s log=%s\n' "$OUT" "$LOG" > "$STATE"
