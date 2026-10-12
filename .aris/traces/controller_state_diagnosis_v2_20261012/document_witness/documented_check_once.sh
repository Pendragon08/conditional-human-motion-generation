#!/bin/bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_v2_20261012.py --check
witness_code=$?
printf 'DOCUMENT_WITNESS_TERMINAL_EXIT=%s\n' "$witness_code"
exit "$witness_code"
