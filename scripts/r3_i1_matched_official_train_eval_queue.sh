#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/third_party/DreamControl/Training"
MON="$ROOT/artifacts/monitoring"
PAIR="$ROOT/artifacts/i1_joint_v2_official_paired_seed42"
MANIFEST="$ROOT/results/r3_i1_joint_v2_paired_seed42_manifest.json"
GATE="$MON/r3_i1_live_training_view_gate.json"
QUEUE="$MON/r3_i1_matched_official_queue.state"
mkdir -p "$MON" "$ROOT/results"
exec 9>"$MON/r3_i1_matched_official_queue.lock"
flock -n 9 || { echo 'Matched queue already running' >&2; exit 1; }

# The user requires a separately monitored, locally visible simulation. The
# visual preflight record must be real; a checkpoint replay is insufficient.
python3 - "$GATE" "$MANIFEST" "$PAIR" <<'PY'
import json, pathlib, sys
gate, manifest, paired = map(pathlib.Path, sys.argv[1:])
if not gate.is_file():
    raise SystemExit("BLOCKED: no local actual-simulation view confirmation")
g = json.loads(gate.read_text())
if g.get("user_confirmed_local_stream") is not True or g.get("mode") != "sampled_actual_training_simulation":
    raise SystemExit("BLOCKED: local view gate not satisfied")
m = json.loads(manifest.read_text())
ids = m["paired_train_ids"]
if len(ids) < 30:
    raise SystemExit("BLOCKED: fewer than 30 paired train references")
for arm in ("baseline", "candidate"):
    files = {int(p.stem) for p in (paired / arm).glob("*.pkl")}
    if files != set(ids):
        raise SystemExit(f"BLOCKED: {arm} reference IDs differ")
print("R3_PAIR_GATE_PASS", len(ids), g["mode"])
PY

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/cache/r3_tmp" XDG_CACHE_HOME="$ROOT/cache/r3_xdg"
export MPLCONFIGDIR="$ROOT/cache/r3_matplotlib" WANDB_MODE=disabled WANDB_SILENT=true
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/scripts:$ROOT/envs/dreamcontrol-r3/site-packages:$TRAIN/source/isaaclab:$TRAIN/source/isaaclab_assets:$TRAIN/source/isaaclab_tasks:$TRAIN/source/isaaclab_rl:$TRAIN/isaac_utils:$TRAIN/scripts/reinforcement_learning/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TRAIN"

for arm in baseline candidate; do
  REF="$PAIR/$arm"
  export DREAMCONTROL_KICK_REF_PATH="$REF"
  LOG="$MON/r3_i1_matched_${arm}_8192_2000.log"
  STATE="$MON/r3_i1_matched_${arm}_8192_2000.state"
  [[ ! -e "$LOG" && ! -e "$STATE" ]] || {
    echo "BLOCKED preexisting_${arm}_run" > "$QUEUE"; exit 1;
  }
  echo "RUNNING arm=$arm refs=$REF envs=8192 iterations=2000 seed=42 original_task=true no_extra_ik=true telemetry_udp=127.0.0.1:40123" > "$STATE"
  echo "RUNNING arm=$arm" > "$QUEUE"
  set +e
  timeout --signal=TERM --kill-after=45s 172800s \
    "$ROOT/third_party/isaacsim-4.5.0/python.sh" -u \
    "$ROOT/scripts/r3_i1_train_matched.py" \
    --task=Isaac-Motion-Tracking-Kick-v0 --device=cuda:1 \
    --num_envs=8192 --max_iterations=2000 --seed=42 --headless \
    --telemetry-udp=127.0.0.1:40123 \
    > "$LOG" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )) || ! grep -q 'Learning iteration 1999/2000' "$LOG"; then
    echo "FAILED arm=$arm exit=$rc log=$LOG" > "$STATE"
    echo "BLOCKED arm=$arm training_failed" > "$QUEUE"
    exit 1
  fi
  run_name="$(sed -n 's/^Exact experiment name requested from command line: //p' "$LOG" | tail -n 1 | tr -d '\r')"
  checkpoint="$TRAIN/logs/rsl_rl/g1/$run_name/model_1999.pt"
  [[ -n "$run_name" && -s "$checkpoint" ]] || {
    echo "FAILED arm=$arm checkpoint_missing run=$run_name" > "$STATE"
    echo "BLOCKED arm=$arm checkpoint_missing" > "$QUEUE"
    exit 1
  }
  sha="$(sha256sum "$checkpoint" | cut -d' ' -f1)"
  echo "COMPLETE arm=$arm iteration=1999/2000 checkpoint=$checkpoint sha256=$sha" > "$STATE"
  echo "EVALUATING arm=$arm" > "$QUEUE"
  bash "$ROOT/scripts/r3_i1_matched_official_eval.sh" "$arm" "$checkpoint" || {
    echo "BLOCKED arm=$arm evaluation_failed" > "$QUEUE"; exit 1;
  }
done
echo 'COMPLETE arms=baseline,candidate envs=8192 iterations=2000 seed=42 matched_official_kick=true' > "$QUEUE"
