"""Exclusive continuation dispatch; no original job restart or signal."""
import argparse,ast,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isascii() and a.run.isidentifier()
expected=hashlib.sha256((B/'configs/user-vae100k-resume-spec.json').read_bytes()).hexdigest()
code=r'''
import fcntl,json,os,pathlib,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;sys.path.insert(0,str(r/'scripts'))
from user_vae100k_resume_protocol import failed_source,read,sha
assert sha(r/'configs/user-vae100k-resume-spec.json')==__SHA__;s=read(r/'configs/user-vae100k-resume-spec.json');assert s['witness_authorized'] and (__MODE__=='witness' or s['train_authorized']);failed_source(s)
prefix='user_vae100k_resume_'+__RUN__;assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_process_exit.json','_exit.json','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
cmd=['python3',str(r/'scripts/supervise_user_vae100k_resume.py'),'--mode',__MODE__,'--run',__RUN__];driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
text="import subprocess,json,pathlib\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f)\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text)
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=r)
z=dict(status='USER_VAE100K_RESUME_DISPATCHED_NOT_YET_ACCEPTED',mode=__MODE__,run=__RUN__,driver_pid=q.pid,spec_sha256=__SHA__)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(z,f,indent=2)
print(json.dumps(z))
'''
for k,v in [('__SHA__',expected),('__MODE__',a.mode),('__RUN__',a.run)]:code=code.replace(k,repr(v))
ast.parse(code);out=B/'results/vae100k_user_handoff_20261028'/a.run;out.mkdir(exist_ok=False)
z=json.loads(recorded_remote(code,out/'dispatch.json',timeout=120));(out/'launch.json').write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z))
