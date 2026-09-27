#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIMIT="${1:-2}"
SEED="${2:-10}"
SUFFIX="n${LIMIT}"
if [[ "$SEED" != 10 ]]; then SUFFIX="${SUFFIX}_seed${SEED}"; fi
STAGE="$ROOT/third_party/DreamControl/TrajGen/sample_r3_${SUFFIX}"
STATE="$ROOT/artifacts/monitoring/r3_dreamcontrol_refine_${SUFFIX}.state"
LOG="$ROOT/artifacts/monitoring/r3_dreamcontrol_refine_${SUFFIX}.log"
input_count="$(find "$STAGE/Kick_sim1" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
if (( input_count < 1 )); then
  printf 'BLOCKED no_retargeted_inputs\n' > "$STATE"
  exit 1
fi
if compgen -G "$STAGE/Kick_sim2/*.pkl" > /dev/null; then
  printf 'BLOCKED output_already_exists\n' > "$STATE"
  exit 1
fi
used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$ROOT/envs/dreamcontrol-retarget/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/envs/dreamcontrol-retarget:$ROOT/third_party/pyroki/src:$ROOT/third_party/jaxls/src:$ROOT/third_party/DreamControl/Training/isaac_utils:$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$STAGE/Kick_sim1"
printf 'RUNNING started=%s input_count=%s gpu=1\n' "$(date -u +%FT%TZ)" "$input_count" > "$STATE"
timeout --signal=TERM --kill-after=30s 1800s \
  "$ROOT/.venv-retarget/bin/python" -u refine_motions.py > "$LOG" 2>&1
result=$?
output_count="$(find "$STAGE/Kick_sim2" -maxdepth 1 -name '*.pkl' -type f | wc -l)"
if (( result != 0 )) || (( output_count < 1 )); then
  printf 'BLOCKED exit=%s output_count=%s log=%s\n' "$result" "$output_count" "$LOG" > "$STATE"
  exit 1
fi
printf 'FINISHED_NEEDS_VALIDATION input_count=%s output_count=%s output=%s\n' "$input_count" "$output_count" "$STAGE/Kick_sim2" > "$STATE"
