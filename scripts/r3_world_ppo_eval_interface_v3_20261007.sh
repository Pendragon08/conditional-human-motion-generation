#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/artifacts/monitoring/r3_world_ppo_eval_interface_v3_20261007"
[[ ! -e "$PREFIX.state" ]] || exit 1
for ARM in P-history W-correct; do
 echo "RUNNING diagnostic_only=true arm=$ARM PPO_updates=0" > "$PREFIX.state"
 set +e
 bash "$ROOT/scripts/r3_world_ppo_formal_stage_v3_20261007.sh" eval-diagnostic "$ARM" 42
 rc=$?
 set -e
 if (( rc )); then echo "FAILED arm=$ARM wrapper_exit=$rc preserve=true" > "$PREFIX.state"; exit 1; fi
done
echo 'COMPLETE P-history_exit=0 W-correct_exit=0 diagnostic=true formal=false' > "$PREFIX.state"
echo WORLD_PPO_EVAL_INTERFACE_QUEUE_COMPLETE
