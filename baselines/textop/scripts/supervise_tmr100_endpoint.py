"""Bound only the CPU observer, without owning the production trainer group."""
import argparse,json,os,pathlib
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from tmr100_endpoint_protocol import CONTROL_RUN,read,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run==CONTROL_RUN and os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sp=ROOT/'configs/tmr100-user-control-spec.json';s=read(sp)
assert s['control_authorized'] and sha(__file__)==s['supervisor_sha256']
assert sha(ROOT/'scripts/watch_tmr100_endpoint.py')==s['entrypoint_sha256']
for name,value in s['bound_sources'].items():assert sha(ROOT/name)==value
prefix='tmr100_control_'+a.run;log=ROOT/'logs';process=ROOT/'receipts'/(prefix+'_process_exit.json')
resources=log/(prefix+'_resources.jsonl');ex=ROOT/'receipts'/(prefix+'_exit.json')
assert not ex.exists()
cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(ROOT/'scripts/watch_tmr100_endpoint.py'),'--run',a.run]
proof=run_supervised(cmd,log/(prefix+'.log'),resources,process,
    lambda pid:sample_resources(ROOT,pid,1024**3,2*1024**3),7*24*3600+360,15)
rp=ROOT/'evaluator_tmr_20261029'/a.run/'receipt.json';receipt=read(rp) if rp.exists() else {}
proof.update(run=a.run,source_sha256=sha(__file__),request_sha256=sha(sp),receipt_sha256=sha(rp) if rp.exists() else None,
    process_exit_sha256=sha(process),resource_sha256=sha(resources),model_lock_acquired=False)
with ex.open('x') as stream:json.dump(proof,stream,indent=2);stream.flush();os.fsync(stream.fileno())
assert proof['supervision_passed'] and receipt['status']=='USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
