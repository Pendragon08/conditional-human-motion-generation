"""CPU-only real filtered user-shortened VAE100k inference export after the training exits."""
import argparse,fcntl,json,os,pathlib,shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from filtered15000_corpus_acceptance import read,sha,owned
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and a.run.isascii() and a.run.isidentifier()
sp=ROOT/'configs/user-vae100k-export-spec.json';s=read(sp)
assert s['schema']=='textop.user_vae100k_cpu_export.v1' and s['export_authorized']
assert sha(__file__)==s['supervisor_sha256'] and sha(ROOT/'scripts/export_user_vae100k.py')==s['executor_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(ROOT).free>=s['minimum_disk_free_bytes']
parent=ROOT/'filtered15000_20261027/vae_export_runs';parent.mkdir(mode=0o700,exist_ok=True);owned(parent,ROOT)
out=parent/a.run;assert not out.exists();prefix='filtered15000_vae_export_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PYTHONIOENCODING=utf-8','PYTHONUNBUFFERED=1',
 str(ROOT/'env/bin/python'),str(ROOT/'scripts/export_user_vae100k.py'),'--run',a.run]
resources=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,
 lambda pid:sample_resources(ROOT,pid,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes']),s['timeout_seconds'],15)
rp=out/'receipt.json';rec=read(rp) if rp.exists() else {}
proof.update(run=a.run,status=rec.get('status'),receipt_sha256=sha(rp) if rp.exists() else None,
 spec_sha256=sha(sp),source_sha256=sha(__file__),resource_sha256=sha(resources),process_exit_sha256=sha(process),
 new_gpu_contexts=0,cpu_model_loads=rec.get('cpu_model_loads',0),new_sdk_runs=0,optimizer_updates=0)
with ep.open('x') as f:json.dump(proof,f,indent=2,allow_nan=False)
print(json.dumps(proof),flush=True)
assert proof['supervision_passed'] and rec.get('status')=='USER_VAE100000_CPU_EXPORT_PASSED' and rec.get('output_writers_closed')
