"""Durable exclusive dispatch for the explicit user100k successors only."""
import argparse,ast,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['quality','export','ldm_witness','ldm_train'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
mapping={'quality':('user-vae100k-quality-spec.json','supervise_user_vae100k_quality.py','user_vae100k_quality_',[]),
 'export':('user-vae100k-export-spec.json','supervise_user_vae100k_export.py','filtered15000_vae_export_',[]),
 'ldm_witness':('user-vae100k-ldm-spec.json','supervise_user_vae100k_ldm.py','filtered15000_ldm_',['--mode','witness']),
 'ldm_train':('user-vae100k-ldm-spec.json','supervise_user_vae100k_ldm.py','filtered15000_ldm_',['--mode','train'])}
name,script,prefix,flags=mapping[a.stage];expected=hashlib.sha256((B/'configs'/name).read_bytes()).hexdigest()
code=r'''
import pathlib,json,os,sys,fcntl,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from user_vae100k_completion import stopped_vae100k,accepted_vae100k,read,sha
stage=__STAGE__;run=__RUN__;prefix=__PREFIX__+run;sp=r/'configs'/__SPEC__;assert sha(sp)==__EXPECTED__;s=read(sp)
assert sha(r/'scripts'/__SCRIPT__)==s['supervisor_sha256']
if stage=='quality':assert s['quality_authorized'];stopped_vae100k(s['vae_training'])
else:
 accepted_vae100k(s['vae_training'])
 if stage=='export':assert s['export_authorized'] and s['export_dir']
 else:assert s['witness_authorized'] and (stage=='ldm_witness' or s['train_authorized'])
paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json','_process_exit.json']]
assert not any(p.exists() for p in paths),'Existing intent/outcome: inspect, never retry blindly'
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
cmd=['python3',str(r/'scripts'/__SCRIPT__)]+__FLAGS__+['--run',run]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,json,pathlib\nq=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':q.returncode},f)\nraise SystemExit(q.returncode)\n"
with driver.open('x') as f:f.write(text)
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='USER_VAE100K_SUCCESSOR_DISPATCHED_NOT_YET_ACCEPTED',stage=stage,run=run,driver_pid=q.pid,spec_sha256=__EXPECTED__,command=cmd)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
for k,v in [('__STAGE__',a.stage),('__RUN__',a.run),('__PREFIX__',prefix),('__SPEC__',name),('__EXPECTED__',expected),('__SCRIPT__',script),('__FLAGS__',flags)]:code=code.replace(k,repr(v))
ast.parse(code);out=B/('results/vae100k_user_handoff_20261028' if a.stage=='quality' else 'results/filtered15000_handoff_20261027')/a.run;out.mkdir(parents=True,exist_ok=False)
j=json.loads(recorded_remote(code,out/'dispatch.json',timeout=120));(out/'launch.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))
