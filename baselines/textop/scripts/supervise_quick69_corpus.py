"""Bound a private corpus build/loader witness using the unchanged warm environment."""
import argparse, fcntl, hashlib, json, os, pathlib, re, subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['build','verify'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
specpath=ROOT/'configs/quick69-corpus-spec.json';spec=json.loads(specpath.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert spec['build_authorized'] is True and spec['quick_training_dispatch_authorized'] is False
assert sha(__file__)==spec['supervisor_sha256'] and sha(ROOT/'scripts/build_quick69_corpus.py')==spec['builder_sha256']
plan=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text())
for n,h in dict(plan['frozen_runtime_files'],**plan['control_sources']).items():assert sha(ROOT/n)==h
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not {str(ROOT/'scripts'/n) for n in ['train_filter_policy.py','evaluate_quick_pilot.py','build_quick69_corpus.py']}.intersection(args),'Existing personal GPU worker'
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
prefix='quick69_corpus_'+a.run;proofpath=ROOT/'receipts'/(prefix+'_exit.json');assert not proofpath.exists()
envargs=['CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','WANDB_MODE=disabled','PYTHONUNBUFFERED=1']
cmd=['taskset','-c','0-3','/usr/bin/env']+envargs+[str(ROOT/'env/bin/python'),str(ROOT/'scripts/build_quick69_corpus.py'),'--spec',str(specpath),'--mode',a.mode,'--run',a.run]
samples=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),samples,process,
 lambda pid:sample_resources(ROOT,pid,2*1024**3,spec['reserve_bytes']),900,15)
rp=ROOT/'quick_generator_20261026/corpus_validation'/a.run/'receipt.json';r=json.loads(rp.read_text()) if rp.exists() else {}
proof.update(mode=a.mode,run=a.run,source_sha256=sha(__file__),spec_sha256=sha(specpath),
             numerical_status=r.get('status'),evaluation_receipt_sha256=sha(rp) if rp.exists() else None,
             process_exit_sha256=sha(process),resources_sha256=sha(samples),new_sdk_runs=0,new_physics_steps=0,optimizer_updates=0)
with proofpath.open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
assert proof['supervision_passed'] and r.get('status')=='QUICK69_CORPUS_LOADER_WITNESS_PASSED' and r.get('output_writers_closed') is True
