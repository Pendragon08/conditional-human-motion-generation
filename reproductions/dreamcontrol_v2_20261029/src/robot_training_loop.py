"""Experimental single-process training loop with resumable sampler cursor.

Caller owns data acceptance, model/diffusion initialization, provenance,
hyperparameter verification, resource checks and directory isolation. No GPU
job is launched by importing this module. Uses zero data-loader workers and
exact-length batches so RNG and cursor can be restored deterministically.
This module is not an assertion of DreamControl-v2 fidelity.
"""
from __future__ import annotations
from pathlib import Path
import os
import random
import numpy as np
import torch
from length_bucket_sampler import ExactLengthBatchSampler
from robot_motion_dataset import collate_robot_motion
from robot_training_step import train_step, train_accumulated_step
from training_checkpoint import load_checkpoint, save_checkpoint


def run_training(model, diffusion, optimizer, dataset, *, run_dir: Path, provenance: dict,
                 config: dict, max_steps: int, checkpoint_every: int,
                 batch_size: int, seed: int, guidance_steps: int, guidance_scale: float,
                 device: torch.device, resume_from: Path | None = None, on_step=None,
                 accumulate_microbatches: int = 1):
    run_dir = Path(run_dir).resolve(strict=True)
    if not run_dir.is_dir(): raise ValueError('run directory must exist')
    lock = run_dir/'training.lock'
    descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(descriptor,str(os.getpid()).encode('ascii'))
        return _run_training(model,diffusion,optimizer,dataset,run_dir=run_dir,provenance=provenance,
                             config=config,max_steps=max_steps,checkpoint_every=checkpoint_every,
                             batch_size=batch_size,seed=seed,guidance_steps=guidance_steps,
                             guidance_scale=guidance_scale,device=device,resume_from=resume_from,on_step=on_step,
                             accumulate_microbatches=accumulate_microbatches)
    finally:
        os.close(descriptor)
        lock.unlink()


def _run_training(model, diffusion, optimizer, dataset, *, run_dir: Path, provenance: dict,
                  config: dict, max_steps: int, checkpoint_every: int,
                  batch_size: int, seed: int, guidance_steps: int, guidance_scale: float,
                  device: torch.device, resume_from: Path | None = None, on_step=None,
                  accumulate_microbatches: int = 1):
    if not isinstance(max_steps,int) or isinstance(max_steps,bool) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if not isinstance(checkpoint_every,int) or isinstance(checkpoint_every,bool) or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    if not isinstance(config,dict) or config.get('batch_size') != batch_size or config.get('seed') != seed:
        raise ValueError('config must record the exact sampler batch size and seed')
    if config.get('guidance_steps') != guidance_steps or config.get('guidance_scale') != guidance_scale:
        raise ValueError('config must record the exact guidance settings')
    if (not isinstance(accumulate_microbatches,int) or isinstance(accumulate_microbatches,bool) or
            not 1 <= accumulate_microbatches <= 64 or
            config.get('accumulate_microbatches',1) != accumulate_microbatches):
        raise ValueError('config must record a valid microbatch accumulation count')
    run_dir = Path(run_dir).resolve(strict=True)
    if not run_dir.is_dir(): raise ValueError('run directory must exist')
    model_parameters = {id(p) for p in model.parameters() if p.requires_grad}
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    if not model_parameters or len(optimizer_params) != len({id(p) for p in optimizer_params}) or {id(p) for p in optimizer_params} != model_parameters:
        raise ValueError('optimizer parameters must exactly match trainable model parameters')
    sampler = ExactLengthBatchSampler(dataset,batch_size,seed=seed,shuffle=True,drop_last=False)
    if len(sampler) < 1: raise ValueError('no training batches')
    if resume_from is not None:
        resume_from = Path(resume_from).resolve(strict=True)
        if resume_from.parent != run_dir:
            raise ValueError('checkpoint must belong to this isolated run directory')
        cursor = load_checkpoint(resume_from,model,optimizer,expected_provenance=provenance,expected_config=config)
    else:
        # Existing checkpoints could belong to another run; never overwrite or merge.
        if any(p.name != 'training.lock' for p in run_dir.iterdir()):
            raise ValueError('new run directory must be empty')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(seed)
        cursor = {'step':0,'epoch':0,'next_batch':0}
    if cursor['next_batch'] >= len(sampler):
        raise ValueError('checkpoint cursor lies outside the current sampler')
    if cursor['step'] > max_steps: raise ValueError('checkpoint is beyond requested max_steps')
    model.train()
    # CLIP is frozen; keep dropout/batch normalization behavior fixed.
    if hasattr(model,'clip_model'): model.clip_model.eval()
    completed_here = 0
    while cursor['step'] < max_steps:
        next_epoch, next_index = cursor['epoch'], cursor['next_batch']
        microbatches = []
        for _ in range(accumulate_microbatches):
            sampler.set_epoch(next_epoch)
            batches = list(sampler)
            motion, cond = collate_robot_motion([dataset[i] for i in batches[next_index]])
            motion = motion.to(device)
            cond['y'] = {k:v.to(device) if torch.is_tensor(v) else v for k,v in cond['y'].items()}
            microbatches.append((motion,cond))
            next_index += 1
            if next_index == len(batches):
                next_epoch += 1
                next_index = 0
                # Flush the partial accumulation at the epoch boundary. Mixing
                # the next shuffle into this update could repeat an example.
                break
        if accumulate_microbatches == 1:
            metrics = train_step(model,diffusion,optimizer,*microbatches[0],
                                 guidance_steps=guidance_steps,guidance_scale=guidance_scale)
        else:
            metrics = train_accumulated_step(model,diffusion,optimizer,microbatches,
                                             guidance_steps=guidance_steps,guidance_scale=guidance_scale)
        cursor = {'step':cursor['step']+1,'epoch':next_epoch,'next_batch':next_index}
        completed_here += 1
        if cursor['step'] % checkpoint_every == 0 or cursor['step'] == max_steps:
            checkpoint = run_dir/f"step_{cursor['step']:09d}.pt"
            budget = config.get('checkpoint_size_estimate')
            guard = {'max_checkpoint_bytes':budget,'disk_floor_bytes':8*1024**3} if budget is not None else {}
            save_checkpoint(checkpoint,model,optimizer,provenance=provenance,config=config,cursor=cursor,**guard)
        if on_step is not None: on_step(metrics,dict(cursor))
    return {'cursor':cursor,'steps_this_call':completed_here,'batch_count_per_epoch':len(sampler)}
