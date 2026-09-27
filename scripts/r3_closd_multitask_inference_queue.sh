#!/usr/bin/env bash
set -uo pipefail

# Launch one bounded, official-code CLoSD inference only after licensed assets
# and the already queued DreamControl GPU-1 interim evaluation are complete.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MON="$ROOT/artifacts/monitoring"
ASSET="$ROOT/artifacts/closd-r3/official_assets_manifest.json"
ASSET_STATE="$MON/r3_closd_asset_download.state"
INTERIM="$MON/r3_resume_interim_1600_gpu1.state"
STATE="$MON/r3_closd_multitask_inference.state"
LOG="$MON/r3_closd_multitask_inference.log"
LOCK="$MON/r3_closd_multitask_inference.lock"
mkdir -p "$MON" "$ROOT/cache/closd-r3-torch-extensions"
exec 9>"$LOCK"
flock -n 9 || exit 0
if [[ -f "$STATE" ]] && grep -q '^FINISHED_NEEDS_VALIDATION ' "$STATE"; then exit 0; fi
printf 'WAITING_FOR_ASSETS started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 960); do
  [[ -s "$ASSET" ]] && break
  if [[ -f "$ASSET_STATE" ]] && grep -q '^BLOCKED ' "$ASSET_STATE"; then
    printf 'BLOCKED asset_download_failed\n' > "$STATE"
    exit 1
  fi
  sleep 30
done
if [[ ! -s "$ASSET" ]]; then
  printf 'BLOCKED asset_wait_timeout\n' > "$STATE"
  exit 1
fi
printf 'WAITING_FOR_GPU1_INTERIM started=%s\n' "$(date -u +%FT%TZ)" > "$STATE"
for _ in $(seq 1 240); do
  if [[ -f "$INTERIM" ]] && grep -Eq '^(COMPLETE|BLOCKED) ' "$INTERIM"; then break; fi
  sleep 30
done
if [[ ! -f "$INTERIM" ]] || ! grep -Eq '^(COMPLETE|BLOCKED) ' "$INTERIM"; then
  printf 'BLOCKED interim_eval_still_pending\n' > "$STATE"
  exit 1
fi
for _ in $(seq 1 120); do
  used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then break; fi
  sleep 30
done
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used >= 1000 )); then
  printf 'BLOCKED gpu1_busy used_mib=%s\n' "$used" > "$STATE"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1 MAX_JOBS=2
export PYTHONPATH="$ROOT/envs/closd-r3/site-packages:$ROOT/third_party/CLoSD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/envs/closd-r3/site-packages/ninja/data/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/closd-r3-torch-extensions"
export HF_HOME="$ROOT/cache/closd_hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled WANDB_SILENT=true
cd "$ROOT/third_party/CLoSD"
printf 'RUNNING started=%s gpu=1 envs=9 games=9 episode_length=300\n' "$(date -u +%FT%TZ)" > "$STATE"
timeout --signal=TERM --kill-after=30s 1200s \
  "$ROOT/.venv-pbhc-training/bin/python" -u closd/run.py \
  learning=im_big robot=smpl_humanoid \
  epoch=-1 test=True no_virtual_display=True \
  headless=True env.num_envs=9 env.episode_length=300 \
  env=closd_multitask exp_name=CLoSD_multitask_finetune \
  learning.params.config.player.games_num=9 \
  > "$LOG" 2>&1
result=$?
if (( result != 0 )) || grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory' "$LOG"; then
  printf 'BLOCKED inference_exit=%s log=%s\n' "$result" "$LOG" > "$STATE"
  exit 1
fi
printf 'FINISHED_NEEDS_VALIDATION exit=0 log=%s\n' "$LOG" > "$STATE"
