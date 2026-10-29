"""Future bounded CPU job; holds the same model lock as the trainer."""
import argparse,fcntl,json,os,pathlib,shutil
from step4_ppo5000_frozen_protocol_v2 import ROOT,RESERVE,LOG,owned,read,sha,accepted_stop,accepted_export
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
sp=owned(a.request);s=read(sp)
assert os.getuid()==1013 and sha(__file__)==s['supervisor_sha256'] and s['authorized'] is True
assert sha(owned(ROOT/'scripts/run_ppo5000_frozen_cpu_v2.py'))==s['executor_sha256']
for n,h in s['bound_sources'].items():assert sha(owned(ROOT/n))==h
os.sched_setaffinity(0,{0,1,2,3})
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
accepted_stop(s['training_binding'])
if s['role']=='verify':accepted_export(s['export_binding'])
parent=owned(ROOT/'step34_numeric_v1');assert not (parent/s['run']).exists()
log=parent/(s['run']+'_supervision');assert not log.exists()
assert shutil.disk_usage(parent).free>=RESERVE+LOG+(32*1024**2 if s['role']=='export' else 0)
log.mkdir(mode=0o700,exist_ok=False)
def sample(pid):
 row=sample_resources(ROOT,pid,16*1024**3,RESERVE+LOG)
 remaining=32*1024**2 if s['role']=='export' and not (parent/s['run']/'actor_RMS.private.pth').exists() else 0
 if remaining:
  temp=parent/s['run']/'actor_RMS.private.pth.partial'
  if temp.exists():remaining=max(0,remaining-temp.stat().st_size)
 assert shutil.disk_usage(parent).free>=RESERVE+LOG+remaining
 return row
cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','bash',str(ROOT/'scripts/run_tracker.sh'),
 str(ROOT/'scripts/run_ppo5000_frozen_cpu_v2.py'),'--request',str(sp)]
proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,900,15)
rp=parent/s['run']/'receipt.json';rec=read(rp) if rp.exists() else {}
proof.update(run=s['run'],source_sha256=sha(__file__),request_sha256=sha(sp),receipt_sha256=sha(rp) if rp.exists() else None,
 resource_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'))
with (log/'exit.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
expected='PPO5000_FROZEN_CPU_EXPORTED' if s['role']=='export' else 'PPO5000_FROZEN_FRESH_CPU_VERIFIED'
assert proof['supervision_passed'] and rec['status']==expected and rec['output_writers_closed']
