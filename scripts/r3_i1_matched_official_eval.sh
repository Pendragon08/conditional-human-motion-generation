#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: eval.sh ARM CHECKPOINT' >&2; exit 2; }
ARM="$1" CKPT="$2"
[[ "$ARM" == baseline || "$ARM" == candidate ]] || exit 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
[[ -s "$CKPT" ]] || exit 1
cd "$TRAIN"
for seed in 43 44; do
  for ref_arm in baseline candidate; do
    REF="$ROOT/artifacts/i1_joint_v2_official_paired_seed$seed/$ref_arm"
    REF_MANIFEST="$ROOT/results/r3_i1_joint_v2_paired_seed${seed}_manifest.json"
    LOG="$MON/r3_i1_matched_${ARM}_eval_seed${seed}_${ref_arm}.log"
    STATE="$MON/r3_i1_matched_${ARM}_eval_seed${seed}_${ref_arm}.state"
    [[ -s "$REF_MANIFEST" && -d "$REF" && ! -e "$LOG" ]] || {
      echo "BLOCKED missing_or_preexisting arm=$ARM seed=$seed ref=$ref_arm" > "$STATE"; exit 1;
    }
    export DREAMCONTROL_KICK_REF_PATH="$REF"
    echo "RUNNING policy=$ARM holdout_seed=$seed refs=$ref_arm envs=1000 frames=500" > "$STATE"
    set +e
    timeout --signal=TERM --kill-after=45s 7200s \
      "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
      scripts/reinforcement_learning/rsl_rl/play_eval.py \
      --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:1 \
      --num_envs=1000 --path="$CKPT" --headless --video --video_length=500 \
      --id="r3_i1_${ARM}_seed${seed}_${ref_arm}" \
      > "$LOG" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )) || ! grep -q 'Success rate:' "$LOG"; then
      echo "FAILED policy=$ARM seed=$seed refs=$ref_arm exit=$rc" > "$STATE"; exit 1
    fi
    success="$(sed -n 's/^Success rate: *\([0-9.]*\) *%.*/\1/p' "$LOG" | tail -n 1)"
    [[ -n "$success" ]] || { echo "FAILED success_parse" > "$STATE"; exit 1; }
    video="$(dirname "$CKPT")/videos/play/eval_a_r3_i1_${ARM}_seed${seed}_${ref_arm}-step-0.mp4"
    [[ -s "$video" ]] || { echo "FAILED video_missing_or_empty policy=$ARM seed=$seed refs=$ref_arm" > "$STATE"; exit 1; }
    video_sha="$(sha256sum "$video" | cut -d' ' -f1)"
    echo "COMPLETE policy=$ARM seed=$seed refs=$ref_arm success_percent=$success video=$video video_sha256=$video_sha" > "$STATE"
  done
done
