#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIMIT="${1:-2}"
SEED="${2:-10}"
SUFFIX="n${LIMIT}"
if [[ "$SEED" != 10 ]]; then SUFFIX="${SUFFIX}_seed${SEED}"; fi
TRAJ="$ROOT/third_party/DreamControl/TrajGen"
SOURCE="$ROOT/artifacts/dreamcontrol-r3/generated/kick_guided_n${LIMIT}_seed${SEED}/results.npy"
STAGE="$TRAJ/sample_r3_${SUFFIX}"
[[ -s "$SOURCE" ]] || { echo "Missing generated human motion: $SOURCE" >&2; exit 1; }
[[ ! -e "$STAGE" ]] || { echo "Stage already exists: $STAGE" >&2; exit 1; }
mkdir -p "$STAGE/Kick_sim" "$STAGE/Kick_sim1" "$STAGE/Kick_sim2"
cp "$TRAJ/sample/Kick_sim/retarget.py" "$STAGE/Kick_sim/retarget.py"
cp "$TRAJ/sample/Kick_sim1/refine_motions.py" "$STAGE/Kick_sim1/refine_motions.py"
cp -a "$TRAJ/sample/retarget_helpers" "$STAGE/retarget_helpers"
ln -s "$SOURCE" "$STAGE/Kick_sim/results.npy"
echo "$STAGE"
