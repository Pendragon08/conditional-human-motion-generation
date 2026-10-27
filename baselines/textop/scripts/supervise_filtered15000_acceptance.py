"""Durable CPU acceptance finalization after real recovery and private D backup."""
import argparse,json,os,pathlib
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from filtered15000_corpus_acceptance import read,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and a.run.isascii() and a.run.isidentifier()
sp=ROOT/'configs/filtered15000-recovery-spec.json';s=read(sp)
assert s['prepare_authorized'] and s['acceptance_authorized']
assert sha(__file__)==s['acceptance_supervisor_sha256'] and sha(ROOT/'scripts/finalize_filtered15000_data.py')==s['finalizer_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
assert sha(ROOT/'configs/filtered15000-vae-spec.json')==s['vae_spec_sha256']
prefix='filtered15000_acceptance_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','PYTHONIOENCODING=utf-8','PYTHONUNBUFFERED=1',
 'python3',str(ROOT/'scripts/finalize_filtered15000_data.py'),'--run',s['recovery_run'],'--expected-vae-spec-sha',s['vae_spec_sha256']]
resources=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,
 lambda pid:sample_resources(ROOT,pid,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes']),s['timeout_seconds'],15)
rp=ROOT/'receipts/filtered15000_vae_witness_activation.json';rec=read(rp) if rp.exists() else {}
proof.update(run=a.run,status=rec.get('status'),receipt_sha256=sha(rp) if rp.exists() else None,
 spec_sha256=sha(sp),source_sha256=sha(__file__),resource_sha256=sha(resources),process_exit_sha256=sha(process),
 new_gpu_contexts=0,new_model_loads=0,new_sdk_runs=0,optimizer_updates=0)
with ep.open('x') as f:json.dump(proof,f,indent=2,allow_nan=False)
print(json.dumps(proof),flush=True)
assert proof['supervision_passed'] and rec.get('status')=='ACTUAL_FILTERED_CORPUS_VAE_WITNESSES_ENABLED_NOT_DISPATCHED' and rec.get('output_writers_closed')
