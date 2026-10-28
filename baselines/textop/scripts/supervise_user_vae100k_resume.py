"""Exclusive real99700 ->100000 continuation with two bounded resume witnesses."""
import argparse,fcntl,json,os,pathlib,subprocess
from user_vae100k_resume_protocol import ROOT,sha,read,failed_source
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
s=read(ROOT/'configs/user-vae100k-resume-spec.json');assert sha(__file__)==s['supervisor_sha256'] and os.getuid()==1013
assert s['witness_authorized'] and (a.mode=='witness' or s['train_authorized'])
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);failed_source(s)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
prefix='user_vae100k_resume_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','WANDB_MODE=disabled','PYTHONUNBUFFERED=1',str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_user_vae100k_resume.py'),'--mode',a.mode,'--run',a.run]
process=ROOT/'receipts'/(prefix+'_process_exit.json');resources=ROOT/'logs'/(prefix+'_resources.jsonl')
j=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,lambda pid:sample_resources(ROOT,pid,16*1024**3,2*1024**3),1800,15)
rp=ROOT/'vae100k_user_handoff_20261028'/a.run/'receipt.json';r=read(rp) if rp.exists() else {}
j.update(source_sha256=sha(__file__),receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),resource_sha256=sha(resources),numerical_status=r.get('status'))
with ep.open('x') as f:json.dump(j,f,indent=2)
assert j['supervision_passed'] and r['status']==('USER_VAE100K_RESUME_WITNESS_PASSED' if a.mode=='witness' else 'USER_VAE100000_RESUME_FINISHED') and r['output_writers_closed'];print(json.dumps(j))
