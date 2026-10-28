"""Own CPU observer only; it does not own or kill the original trainer group."""
import argparse,hashlib,json,os,pathlib
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run=='user_vae100k_control_v1'
assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
sp=ROOT/'configs/vae100k-user-control-spec.json';s=json.loads(sp.read_bytes())
assert s['control_authorized'] and sha(__file__)==s['supervisor_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h
assert sha(ROOT/'scripts/watch_vae100k_handoff.py')==s['entrypoint_sha256']
prefix='vae100k_control_'+a.run;out=ROOT/'receipts'/(prefix+'_exit.json');assert not out.exists()
cmd=['taskset','-c','0-3','python3',str(ROOT/'scripts/watch_vae100k_handoff.py'),'--run',a.run]
process=ROOT/'receipts'/(prefix+'_process_exit.json');resources=ROOT/'logs'/(prefix+'_resources.jsonl')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,
 lambda pid:sample_resources(ROOT,pid,1024**3,2*1024**3),6*3600+300,15)
rp=ROOT/'vae100k_user_handoff_20261028'/a.run/'receipt.json';r=json.loads(rp.read_bytes()) if rp.exists() else {}
proof.update(run=a.run,source_sha256=sha(__file__),receipt_sha256=sha(rp) if rp.exists() else None,
 process_exit_sha256=sha(process),resource_sha256=sha(resources),numerical_status=r.get('status'),new_gpu_contexts=0)
with out.open('x') as f:json.dump(proof,f,indent=2)
assert proof['supervision_passed'] and r['status']=='USER_VAE100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
print(json.dumps(proof))
