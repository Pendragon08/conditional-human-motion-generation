"""Launch one authorized formal filter run after immutable activation and proofs."""
import json,pathlib,sys,hashlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
A=B/'configs/formal-contact-activation.json'
a=json.loads(A.read_text(encoding='utf-8'))
assert a['launch_enabled'] is True
run='filter_policy_paper69_solid_floor_20261026_v1';assert a['run']==run
for name,wanted in a['deployed_files'].items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==wanted,name
script=r"""
import pathlib,hashlib,json,os,subprocess,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
a=ACTIVATION
assert os.getuid()==1013
for name,wanted in a['deployed_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted,name
plan=json.loads((root/'configs/filter-training-plan.json').read_text())
assert plan['formal_launch_enabled'] is True and plan['formal_launch_blockers']==[]
run=a['run'];out=root/'tracker_validation'/run;assert not out.exists()
proof=root/'receipts'/('formal_dispatch_'+run+'.json')
assert not proof.exists()
# Atomically reserve this exact run; failures remain visible and are not retried.
with proof.open('x') as f:f.write(json.dumps({'run':run,'status':'DISPATCH_RESERVED','time':time.time()}))
log=root/'logs'/('formal_supervisor_'+run+'.log')
command=['python3',str(root/'scripts/supervise_filter_policy.py'),'--run',run,'--mode','formal']
with log.open('xb') as stream:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
r={'run':run,'status':'SUPERVISOR_DISPATCHED_NOT_YET_VERIFIED','supervisor_pid':child.pid,'command':command,'time':time.time(),'deployed_files':a['deployed_files']}
proof.write_text(json.dumps(r,indent=2));print(json.dumps(r))
""".replace('ACTIVATION',repr(a))
r=json.loads(remote(script,timeout=60))
out=B/'results/full_contact_20261026/formal_launch';out.mkdir(exist_ok=True)
(out/'dispatch.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r))
