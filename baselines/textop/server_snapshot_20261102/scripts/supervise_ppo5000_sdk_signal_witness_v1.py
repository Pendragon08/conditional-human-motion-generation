"""Bounded single isolated SDK signal witness supervisor; no training."""
import argparse,fcntl,os,shutil
from step4_train_only_protocol import ROOT,RESERVE,LOG,owned,read,sha,atomic
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
sp=owned(a.request);s=read(sp)
assert os.getuid()==1013 and s['authorized'] is True and s['role']=='isolated_sdk_signal_witness'
assert sha(__file__)==s['supervisor_sha256']
for name,digest in s['bound_sources'].items():assert sha(owned(ROOT/name))==digest
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
parent=owned(s['output_root']);assert not (parent/s['run']).exists()
assert shutil.disk_usage(parent).free>=RESERVE+LOG+s['output_upper_bytes']
log=parent/(s['run']+'_supervision');log.mkdir(mode=0o700,exist_ok=False)
def sample(pid):return sample_resources(ROOT,pid,16*1024**3,RESERVE+LOG)
cmd=['taskset','-c','0-3','/usr/bin/env','PYTHONDONTWRITEBYTECODE=1','PYTHONIOENCODING=utf-8',
 'OMP_NUM_THREADS=4','bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/ppo5000_sdk_signal_witness_v1.py'),
 '--request',str(sp),'--headless','--device','cuda:1']
proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,900,15,grace_seconds=30)
rp=parent/s['run']/'receipt.json';rec=read(rp) if rp.exists() else {}
proof.update(run=s['run'],source_sha256=sha(__file__),request_sha256=sha(sp),
 receipt_sha256=sha(rp) if rp.exists() else None,resource_sha256=sha(log/'resources.jsonl'),
 process_exit_sha256=sha(log/'process_exit.json'),status=rec.get('status'))
atomic(log/'exit.json',proof)
assert proof['supervision_passed'] and proof['child_exit_code']==0
assert rec['output_writers_closed'] and rec['status']=='ISOLATED_SDK_SIGNAL_WITNESS_FINISHED'
assert rec['self_signals']==2 and rec['production_signals']==rec['formal_updates']==rec['optimizer_calls']==0
