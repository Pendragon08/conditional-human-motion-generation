#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code/scripts/train_full.py --config /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code/configs/full_training_v1.json --output /home/ouyanghao/reactivebfm-full-20261021/training_v1 > /home/ouyanghao/reactivebfm-full-20261021/training_v1/stdout.txt 2> /home/ouyanghao/reactivebfm-full-20261021/training_v1/stderr.txt
task_training_exit=$?
printf '%s\n' "$task_training_exit" > /home/ouyanghao/reactivebfm-full-20261021/training_v1/launcher_exit_code.txt
exit "$task_training_exit"
