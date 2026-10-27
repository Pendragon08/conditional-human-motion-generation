"""One CPU audit of real closed replay evidence, serial with personal SDK work."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from audit_teacher15000_full_result import load_closed_run
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and a.run.isascii() and a.run.isidentifier()
sp=ROOT/'configs/teacher15000-full-result-audit-spec.json';s=json.loads(sp.read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert s['schema']=='textop.closed_direct_strict15000_audit_spec.v1' and s['launch_enabled'] is True
assert sha(__file__)==s['supervisor_sha256'] and sha(ROOT/'scripts/audit_teacher15000_full_result.py')==s['executor_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
load_closed_run()  # Refuse before actual full numerical completion and all exits.
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(ROOT).free>=s['minimum_disk_free_bytes']
parent=ROOT/'filtered15000_20261027';parent.mkdir(mode=0o700,exist_ok=True)
assert parent.resolve()==parent and parent.stat().st_uid==1013
out=parent/a.run;assert not out.exists()
prefix='teacher15000_full_audit_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
cmd=['taskset','-c','0-3','/usr/bin/env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PYTHONUNBUFFERED=1','PYTHONIOENCODING=utf-8',
 'python3',str(ROOT/'scripts/audit_teacher15000_full_result.py'),'--output',str(out)]
resources=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),resources,process,
 lambda pid:sample_resources(ROOT,pid,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes']),s['timeout_seconds'],15)
rp=out/'audit.json';result=json.loads(rp.read_text()) if rp.exists() else {}
proof.update(schema='textop.closed_direct_strict15000_audit_supervision.v1',run=a.run,spec_sha256=sha(sp),
 audit_receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),resources_sha256=sha(resources),
 numerical_status=result.get('status'),new_gpu_contexts=0,new_sdk_runs=0,new_physics_steps=0,optimizer_updates=0)
with ep.open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2,allow_nan=False)
print(json.dumps(proof),flush=True)
assert proof['supervision_passed'] and proof['child_exit_code']==0 and result.get('status')=='CLOSED_FULL_STRICT_RESULT_CPU_AUDIT_PASSED'
