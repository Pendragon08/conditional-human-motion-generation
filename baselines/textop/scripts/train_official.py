"""Run unchanged official CLI with resource limits and honest optimizer accounting.

No loss/model/data/sampling algorithm changes. Extra early checkpoint at 1000,
official gradient-skip accounting, disk guard, and JSON metrics are added.
"""
import json,os,pathlib,shutil,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
import torch
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
from robotmdar.train.manager import MVAEManager
start=time.time();updates=0;last=0;skipped=0;consecutive_skips=0
real_step=torch.optim.Adam.step
def counted_step(self,*args,**kwargs):
    global updates
    out=real_step(self,*args,**kwargs);updates+=1;return out
torch.optim.Adam.step=counted_step
original_post=MVAEManager.post_step
def audited_post(self,*args,**kwargs):
    global last,skipped,consecutive_skips
    is_eval=kwargs.get('is_eval',False)
    original_post(self,*args,**kwargs)
    if is_eval:return
    loss=kwargs.get('loss_dict',{}).get('total')
    did_skip=updates==last
    if did_skip:skipped+=1;consecutive_skips+=1
    else:consecutive_skips=0
    metrics={'manager_step':self.step,'optimizer_updates_this_process':updates,'skipped_updates_this_process':skipped,'loss':float(loss),'stage':self.stage_idx,'elapsed_s':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'data_free_bytes':shutil.disk_usage(ROOT).free,'save_dir':str(self.save_dir)}
    if did_skip:
        event={**metrics,'event':'official_nonfinite_gradient_skip','loss_terms':{k:float(v) for k,v in kwargs.get('loss_dict',{}).items()},'nonfinite_gradient_parameters':[name for name,p in self.vae.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]}
        with (ROOT/'logs/gradient_skips.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print('TEXTOP_OFFICIAL_GRADIENT_SKIP',json.dumps(event),flush=True)
        if consecutive_skips>=20:
            self.save_model()
            (ROOT/'receipts/numerical_failure.json').write_text(json.dumps(event,indent=2))
            raise RuntimeError('20 consecutive skipped updates; saved checkpoint for diagnosis')
    last=updates
    if self.step<=10 or self.step%100==0:
        with (ROOT/'logs/mvae_metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        (ROOT/'receipts/training_progress.json').write_text(json.dumps(metrics,indent=2))
        print('TEXTOP_ACTUAL_TRAINING',json.dumps(metrics),flush=True)
    if self.step==1000:self.save_model()
    if metrics['data_free_bytes']<1024**3:
        self.save_model()
        (ROOT/'receipts/disk_guard_stop.json').write_text(json.dumps(metrics,indent=2))
        raise RuntimeError('Stopped after checkpoint save: less than 1GiB disk reserve')
MVAEManager.post_step=audited_post
if __name__=='__main__':
    try:
        from robotmdar.cli import main
        main()
    except BaseException:
        (ROOT/'receipts/training_failure.txt').write_text(traceback.format_exc());raise
