"""One new100k quality process; owns the common lock only after original exit."""
import argparse,fcntl,json,os,pathlib,subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from user_vae100k_completion import ROOT,sha,read,stopped_vae100k
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run=='user_vae100k_quality_v1' and os.getuid()==1013
s=read(ROOT/'configs/user-vae100k-quality-spec.json');assert s['quality_authorized'] and sha(__file__)==s['supervisor_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stopped_vae100k(s['vae_training'])
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
prefix='user_vae100k_quality_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
env=['CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','WANDB_MODE=disabled','PYTHONUNBUFFERED=1']
cmd=['taskset','-c','0-3','/usr/bin/env']+env+[str(ROOT/'env/bin/python'),str(ROOT/'scripts/validate_user_vae100k.py'),'--run',a.run]
process=ROOT/'receipts'/(prefix+'_process_exit.json');resources=ROOT/'logs'/(prefix+'_resources.jsonl')
j=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,lambda pid:sample_resources(ROOT,pid,16*1024**3,2*1024**3),1800,15)
rp=ROOT/'vae100k_user_handoff_20261028'/a.run/'receipt.json';r=read(rp) if rp.exists() else {}
j.update(source_sha256=sha(__file__),receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),resource_sha256=sha(resources),numerical_status=r.get('status'))
with ep.open('x') as f:json.dump(j,f,indent=2)
assert j['supervision_passed'] and r['status']=='USER_VAE100K_QUALITY_VALIDATION_PASSED' and r['output_writers_closed'];print(json.dumps(j))
