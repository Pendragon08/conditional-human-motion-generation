"""Resource-bound the CPU observer only; its signals never target the trainer."""
import argparse,json,os,pathlib
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ppo5000_endpoint_protocol import CONTROL_RUN,read,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
s=read(a.spec);assert os.getuid()==1013 and s['authorized'] and s['run']==CONTROL_RUN
assert sha(__file__)==s['supervisor_sha256'] and sha(ROOT/'scripts/watch_ppo5000_endpoint.py')==s['watcher_sha256']
os.sched_setaffinity(0,{0,1,2,3});prefix='ppo5000_control_'+CONTROL_RUN
logs=ROOT/'logs';process=ROOT/'receipts'/(prefix+'_process_exit.json');resources=logs/(prefix+'_resources.jsonl');ex=ROOT/'receipts'/(prefix+'_exit.json')
assert not ex.exists()
cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(ROOT/'scripts/watch_ppo5000_endpoint.py'),'--spec',a.spec]
proof=run_supervised(cmd,logs/(prefix+'.log'),resources,process,
 lambda pid:sample_resources(ROOT,pid,1024**3,2*1024**3+128*1024**2),72*3600+360,15)
rp=ROOT/'step34_numeric_v1'/CONTROL_RUN/'receipt.json';rec=read(rp) if rp.exists() else {}
proof.update(run=CONTROL_RUN,source_sha256=sha(__file__),spec_sha256=sha(a.spec),
 receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),
 resource_sha256=sha(resources),model_lock_acquired=False)
with ex.open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert proof['supervision_passed'] and rec['status']=='PPO5000_SIGNAL_AND_SAVED_STATE_AWAITING_CPU_AUDIT'
