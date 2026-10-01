#!/usr/bin/env bash
set -u
ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation
TASK=$1
SHARD=$2
DEVICE=$3
NAME=r3_mujoco_robust_contact_v2_${TASK}_shard${SHARD}
mkdir -p "$ROOT/artifacts/monitoring"
exec 9>"$ROOT/artifacts/monitoring/${NAME}.lock"
flock -n 9 || exit 73
STATE="$ROOT/artifacts/monitoring/${NAME}.state"
LOG="$ROOT/artifacts/monitoring/${NAME}.log"
if [[ -e "$STATE" || -e "$ROOT/results/mujoco_robust_contact/${TASK}_shard_${SHARD}.json" ]]; then
  echo "Refusing duplicate worker: $NAME"
  exit 74
fi
printf 'RUNNING task=%s shard=%s started=%s\n' "$TASK" "$SHARD" "$(date -u +%FT%TZ)" > "$STATE"
bash "$ROOT/scripts/r3_mujoco_robust_contact.sh" --mode eval --task "$TASK" --shard "$SHARD" --shards 2 --device "$DEVICE" > "$LOG" 2>&1
RC=$?
if [[ $RC == 0 ]]; then STATUS=COMPLETE; else STATUS=FAILED; fi
printf '%s task=%s shard=%s exit=%s ended=%s\n' "$STATUS" "$TASK" "$SHARD" "$RC" "$(date -u +%FT%TZ)" > "$STATE"
exit "$RC"
