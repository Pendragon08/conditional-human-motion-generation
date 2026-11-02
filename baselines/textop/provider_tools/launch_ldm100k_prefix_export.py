"""One durable CPU export dispatch after actual user100k endpoint acceptance."""
import argparse
import ast
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
sp = B / 'configs/ldm100k-prefix-export-spec.json'
s = json.loads(sp.read_bytes())
assert s['export_authorized'] and s['ldm_training']
expected = hashlib.sha256(sp.read_bytes()).hexdigest()
code = r'''
import fcntl,json,os,pathlib,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_prefix_completion import read,sha,stopped_ldm100k
from contact_filter_resources import sample_resources
run=RUN;prefix='ldm100k_prefix_export_'+run;sp=r/'configs/ldm100k-prefix-export-spec.json'
assert sha(sp)==EXPECTED;s=read(sp);assert s['export_authorized'];stopped_ldm100k(s['ldm_training'])
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
assert sha(r/'scripts/export_ldm100k_prefix.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_prefix_export.py')==s['supervisor_sha256']
paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json','_process_exit.json']]
assert not any(p.exists() for p in paths),'Existing outcome/intent, inspect rather than redispatch'
assert not (r/'ldm100k_user_handoff_20261028'/run).exists()
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes']);lock.close()
cmd=['python3',str(r/'scripts/supervise_ldm100k_prefix_export.py'),'--run',run]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,json,pathlib\nq=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':q.returncode},f)\nraise SystemExit(q.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='LDM100K_PREFIX_CPU_EXPORT_DISPATCHED_NOT_YET_ACCEPTED',run=run,driver_pid=q.pid,spec_sha256=EXPECTED,command=cmd,preflight_resources=resources)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''.replace('RUN', repr(a.run)).replace('EXPECTED', repr(expected))
ast.parse(code, feature_version=(3, 8))
out = B / 'results/ldm100k_user_handoff_20261028' / a.run
out.mkdir(parents=True, exist_ok=False)
j = json.loads(recorded_remote(code, out / 'dispatch.json', timeout=120))
(out / 'launch.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
