"""Resource-bound VAE witness/train dispatcher; serial with personal SDK work."""
import argparse,fcntl,hashlib,json,os,pathlib,re,subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
sp=ROOT/'configs/generator8278-vae-spec.json';spec=json.loads(sp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(__file__)==spec['supervisor_sha256'] and sha(ROOT/'scripts/run_generator8278_vae.py')==spec['entrypoint_sha256']
assert spec['witness_authorized'] and (a.mode=='witness' or spec['train_authorized'])
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text())
for n,h in dict(plan['frozen_runtime_files'],**plan['control_sources']).items():assert sha(ROOT/n)==h
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
prefix='generator8278_vae_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
env=['CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','WANDB_MODE=disabled','PYTHONUNBUFFERED=1']
cmd=['taskset','-c','0-3','/usr/bin/env']+env+[str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_generator8278_vae.py'),'--mode',a.mode,'--run',a.run]
samples=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),samples,process,
 lambda pid:sample_resources(ROOT,pid,16*1024**3,2*1024**3),1800 if a.mode=='witness' else 7*24*3600,15)
rp=ROOT/'generator8278_20261027/vae_runs'/a.run/'receipt.json';r=json.loads(rp.read_text()) if rp.exists() else {}
proof.update(run=a.run,mode=a.mode,numerical_status=r.get('status'),evaluation_receipt_sha256=sha(rp) if rp.exists() else None,
             process_exit_sha256=sha(process),resources_sha256=sha(samples),new_sdk_runs=0,new_physics_steps=0)
with ep.open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof));expected='GENERATOR8278_VAE_WITNESS_PASSED' if a.mode=='witness' else 'GENERATOR8278_VAE_200000_TRAINING_FINISHED'
assert proof['supervision_passed'] and r.get('status')==expected and r.get('output_writers_closed') is True
