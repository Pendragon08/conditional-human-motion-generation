"""Instrument the unmodified official training entrypoint for a bounded real-data pilot."""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'upstream'))


def weight_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def execute(a,state):
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25,0)
    spec=json.loads((BASE/'configs/env_spec.json').read_text())
    if os.statvfs(a.runtime_root).f_bavail*os.statvfs(a.runtime_root).f_frsize < 3*1024**3+1500000000:
        raise RuntimeError('Not enough disk for a single full-architecture checkpoint and reserve')
    from reactivebfm.train import run_utils
    from reactivebfm.train.logging_platforms import NoPlatform
    from reactivebfm.train.scheduled_forcing_loop import ScheduledForcingLoop
    from reactivebfm.data.datasets import get_dataset_loader
    from reactivebfm.train.train_planner_scheduled_forcing import main
    class LocalPlatform(NoPlatform):
        def report_scalars(self,values,iteration,group_name=None):
            row={'official_step':iteration,'optimizer_updates_seen':state['optimizer_updates'],
                 **{k:float(v) for k,v in values.items()}}
            with (a.output/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    run_utils.TRAIN_PLATFORMS['NoPlatform']=LocalPlatform
    original_init=ScheduledForcingLoop.__init__;original_run=ScheduledForcingLoop.run_loop
    original_where=torch.where
    def observed_where(*args,**kwargs):
        if len(args)==3 and isinstance(args[1],torch.Tensor) and args[1].ndim==4 and tuple(args[1].shape[1:])==(36,1,25):
            if sys._getframe(1).f_code.co_filename.endswith('scheduled_forcing_loop.py'):
                state['model_prefix_samples_selected']+=int(args[0].sum())
                state['prefix_selection_calls']+=1
        return original_where(*args,**kwargs)
    torch.where=observed_where
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        state['trainable_parameters']=sum(p.numel() for p in self.model.parameters_wo_clip())
        state['initial_trainable_sha256']=weight_hash(self.model)
        def after_step(opt,args,kwargs):state['optimizer_updates']+=1
        self.opt.register_step_post_hook(after_step)
    ScheduledForcingLoop.__init__=initialize
    def heldout(loop,batches):
        was_training=loop.model.training;loop.model.eval();values=[]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(4242);torch.cuda.manual_seed_all(4242)
            with torch.no_grad():
                for motion,cond in batches:
                    motion=motion.to(loop.device);bs=len(motion)
                    y={'prefix':motion[...,:25],'mask':torch.ones(bs,1,1,50,device=loop.device,dtype=torch.bool),
                       'text':cond['y']['text']}
                    t,_=loop.schedule_sampler.sample(bs,loop.device)
                    loss=loop.diffusion.training_losses(loop.model,motion[...,25:75],t,
                        model_kwargs={'y':y},dataset=loop.data.dataset)['loss']
                    values.extend(loss.detach().cpu().reshape(-1).tolist())
        loop.model.train(was_training)
        if len(values)!=8 or not np.isfinite(values).all():raise ValueError('Heldout objective check invalid')
        return {'mean':float(np.mean(values)),'per_sample':values,'samples':len(values),
                'noise_seed':4242,'protocol':'teacher-forced single 50-frame primitive, official flow objective; not a motion-quality score'}
    def run(self):
        py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state();cu=torch.cuda.get_rng_state_all()
        loader=get_dataset_loader('kungfu_g1_36dim',batch_size=2,data_dir=str(a.runtime_root/'data/kungfu_g1_36dim'),
            split='test',fixed_len=175,pred_len=50,device='cpu',collate_mode='self_rollout',num_workers=0,
            shuffle=False,deterministic=True)
        batches=list(itertools.islice(loader,4))
        state['heldout_before']=heldout(self,batches)
        random.setstate(py);np.random.set_state(nr);torch.set_rng_state(tr);torch.cuda.set_rng_state_all(cu)
        original_run(self)
        state['heldout_after']=heldout(self,batches)
        state['final_trainable_sha256']=weight_hash(self.model)
        state['weights_changed']=state['initial_trainable_sha256']!=state['final_trainable_sha256']
        state['official_final_step_counter']=self.total_step()
    ScheduledForcingLoop.run_loop=run
    sys.argv=['bounded_official_training','--save_dir',str(a.output/'checkpoints'),
        '--dataset','kungfu_g1_36dim','--data_dir',str(a.runtime_root/'data/kungfu_g1_36dim'),
        '--model_type','flow','--text_encoder',spec['weight_dirs']['text_encoder']['path'],
        '--num_steps','101','--num_warmup_steps','25','--batch_size_local','2',
        '--seed','42','--device','0','--num_workers','0','--train_platform_type','NoPlatform',
        '--save_interval','100','--log_interval','1','--cross_prob','0.1','--no-use_ema']
    state['official_cli_arguments']=sys.argv[1:]
    try:main()
    finally:
        torch.where=original_where;ScheduledForcingLoop.__init__=original_init;ScheduledForcingLoop.run_loop=original_run
    if state['optimizer_updates']!=101 or not state['weights_changed'] or state['model_prefix_samples_selected']<=0:
        raise ValueError('Pilot execution contract did not hold')
    state['status']='LIMITED_OFFICIAL_TRAINING_PASSED'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'optimizer_updates':0,'model_prefix_samples_selected':0,'prefix_selection_calls':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_source_commit':'5a79d03cab824eb5917271ba720294e8a1b9d374',
        'paper_benchmark_reproduced':False,'physics_control_evaluated':False,'full_generator_training_completed':False,
        'instrumentation':'optimizer post-hook, metric sink and count-only torch.where wrapper; official sources unchanged'}
    start=time.monotonic();code=0
    try:execute(a,state)
    except Exception as exc:code=1;state.update({'status':'FAILED','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()})
    state.update({'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
        'exit_code':code,'max_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        'max_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
    (a.output/'run_receipt.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True);sys.exit(code)
