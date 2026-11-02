"""Dispatch one author/fresh witness or the complete declared diagnostic exactly once."""
import argparse
import ast
import hashlib
import json
import pathlib
import sys
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_generation_acceptance import RUNS
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--mode', choices=['witness', 'diagnostic'], required=True)
a = p.parse_args()
assert a.run in {'witness': RUNS,
                 'diagnostic': ['user_ldm100k_diagnostic32_v1']}[a.mode]
sp = B / 'configs/ldm100k-generation-spec.json'
s = json.loads(sp.read_bytes())
assert s['witnesses_authorized'] if a.mode == 'witness' else s['generation_authorized']
code = r'''
import fcntl,json,os,pathlib,subprocess,sys,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned
from ldm100k_scope_acceptance import accepted_scope
from ldm100k_generation_acceptance import accepted_generation_witnesses,RUNS
from ldm100k_generation_artifacts import audit_witness_artifacts
from contact_filter_resources import sample_resources
run=__RUN_VALUE__;mode=__MODE_VALUE__;prefix='ldm100k_generation_'+run;sp=owned(r/'configs/ldm100k-generation-spec.json')
assert sha(sp)==__EXPECTED_VALUE__;s=read(sp);assert s['witnesses_authorized'] if mode=='witness' else s['generation_authorized']
for n,h in s['bound_sources'].items():assert sha(owned(r/n))==h,n
assert sha(r/'scripts/run_ldm100k_heldout_generation.py')==s['executor_sha256']
assert sha(r/'scripts/supervise_ldm100k_heldout_generation.py')==s['supervisor_sha256']
paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json','_process_exit.json']]
assert not any(p.exists() for p in paths),'Existing intent/outcome; inspect, never redispatch'
assert not (r/'ldm100k_user_handoff_20261028'/run).exists()
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
actual=accepted_scope(s['scope_binding'])
if mode=='diagnostic':
 accepted_generation_witnesses(s['witness_bindings'])
 for name in RUNS:
  out=owned(r/'ldm100k_user_handoff_20261028'/name);audit_witness_artifacts(out,read(out/'receipt.json'))
resources=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes'])
assert shutil.disk_usage(r).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['accepted_output_upper_bytes'][mode]
gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip())*1024**2>=20*1024**3
lock.close();cmd=['python3',str(r/'scripts/supervise_ldm100k_heldout_generation.py'),'--run',run,'--mode',mode]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,json,pathlib\nq=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':q.returncode},f)\nraise SystemExit(q.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='LDM100K_GENERATION_DISPATCHED_NOT_ACCEPTED',run=run,mode=mode,driver_pid=q.pid,spec_sha256=__EXPECTED_VALUE__,command=cmd,preflight_resources=resources)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''.replace('__RUN_VALUE__', repr(a.run)).replace('__MODE_VALUE__', repr(a.mode)).replace('__EXPECTED_VALUE__', repr(hashlib.sha256(sp.read_bytes()).hexdigest()))
ast.parse(code, feature_version=(3, 8))
out = B / 'results/ldm100k_user_handoff_20261028' / a.run
out.mkdir(exist_ok=False)
j = json.loads(recorded_remote(code, out / 'dispatch.json', timeout=180))
(out / 'launch.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
