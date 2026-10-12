#!/bin/bash
set -o pipefail
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation || exit 91
export PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u scripts/task_matched_feedback_continue_20261012.py --run 2>&1 | tee results/task_matched_feedback_continue_20261012/runner.log
exit ${PIPESTATUS[0]}
