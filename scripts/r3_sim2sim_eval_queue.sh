#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${1:?Kick or Punch}"
VERSION="${2:-initial}"
[[ "$VERSION" == initial || "$VERSION" == exact ]] || exit 2
[[ "$TASK" == Kick || "$TASK" == Punch ]] || exit 2
MON="$ROOT/artifacts/monitoring"
mkdir -p "$MON"
exec 9>"$MON/r3_sim2sim_${TASK}_${VERSION}.lock"
flock -n 9 || exit 1
STATE="$MON/r3_sim2sim_${TASK}_${VERSION}.state"
LOG="$MON/r3_sim2sim_${TASK}_${VERSION}.log"
[[ ! -e "$STATE" ]] || { echo 'Evaluation state exists; do not duplicate' >&2; exit 1; }
if python3 - "$ROOT" "$TASK" <<'PY'
import hashlib, json, pathlib, sys
root, task = pathlib.Path(sys.argv[1]), sys.argv[2]
p = root / 'results/sim2sim_mujoco' / (task + '_zero_shot.json')
if not p.is_file(): raise SystemExit(1)
data = json.loads(p.read_text())
current = hashlib.sha256((root / 'scripts/r3_sim2sim_mujoco.py').read_bytes()).hexdigest()
raise SystemExit(0 if data.get('status') == 'COMPLETE' and data.get('provenance', {}).get('adapter_sha256') == current else 1)
PY
then
  echo 'Already completed for this exact adapter; no duplicate run.'
  exit 0
fi
echo "RUNNING task=$TASK frozen_policy=true no_learning=true physics=mujoco" > "$STATE"
set +e
bash "$ROOT/scripts/r3_sim2sim_mujoco.sh" eval "$TASK" > "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )) || ! grep -q "SIM2SIM_COMPLETE $TASK" "$LOG"; then
  echo "FAILED task=$TASK exit=$rc log=$LOG" > "$STATE"
  exit 1
fi
echo "COMPLETE task=$TASK exit=0 result=$ROOT/results/sim2sim_mujoco/${TASK}_zero_shot.json" > "$STATE"
