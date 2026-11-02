"""Serial, bounded pre-filter corpus construction; no optimizer or SDK."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['build','verify'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and a.run.isascii() and a.run.isidentifier()
sp=ROOT/'configs/generator8278-corpus-spec.json';s=json.loads(sp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'scripts/build_generator8278_corpus.py')==s['builder_sha256'] and s['build_authorized']
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
out=pathlib.Path(s['output_root'])/a.run;out.parent.mkdir(parents=True,mode=0o700,exist_ok=True);assert not out.exists()
prefix='generator8278_corpus_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PYTHONUNBUFFERED=1','PYTHONIOENCODING=utf-8',str(ROOT/'env/bin/python'),str(ROOT/'scripts/build_generator8278_corpus.py'),'--spec',str(sp),'--base-spec',str(ROOT/'configs/paper69-env-spec.json'),'--output',str(out)]
if a.mode=='build':cmd.append('--build')
resources=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
e=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,lambda pid:sample_resources(ROOT,pid,16*1024**3,2*1024**3),s['timeout_seconds'],15)
rp=out/'receipt.json';r=json.loads(rp.read_text()) if rp.exists() else {}
e.update(spec_sha256=sha(sp),source_sha256=sha(__file__),run=a.run,mode=a.mode,receipt_sha256=sha(rp) if rp.exists() else None,resource_sha256=sha(resources),process_exit_sha256=sha(process),status=r.get('status'),optimizer_updates=0,new_sdk_runs=0)
with ep.open('x') as f:json.dump(e,f,indent=2)
assert e['supervision_passed'] and e['child_exit_code']==0 and r.get('status')=='GENERATOR8278_CORPUS_LOADER_WITNESS_PASSED' and r.get('output_writers_closed')
print(json.dumps(e),flush=True)
