"""Durable single invocation of an already staged, gated LDM profile."""
import argparse,ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isascii() and a.run.isidentifier()
driver="""import subprocess,pathlib,json,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
z=subprocess.run(['python3',str(r/'scripts/supervise_quick_ldm.py'),'--mode',MODE,'--run',RUN],capture_output=True,text=True,timeout=TIMEOUT)
(r/'receipts'/(RUN+'_supervisor_execution.json')).write_text(json.dumps(dict(exit_code=z.returncode,stdout=z.stdout,stderr=z.stderr,time_unix=time.time()),indent=2))
""".replace('MODE',repr(a.mode)).replace('RUN',repr(a.run)).replace('TIMEOUT','1050' if a.mode=='witness' else '7500')
ast.parse(driver)
script=r'''
import ast,base64,fcntl,json,pathlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;mode=MODE
sp=json.loads((r/'configs/quick-ldm-runtime-spec.json').read_text());assert sp['witness_authorized'] and (mode=='witness' or sp['train_authorized'])
assert not(r/'quick_generator_20261026/ldm_runs'/run).exists() and not(r/'receipts'/('quick_ldm_'+run+'_exit.json')).exists()
lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lk,fcntl.LOCK_UN)
driver=r/'receipts'/(run+'_driver.py');assert not driver.exists();raw=base64.b64decode(DRIVER);ast.parse(raw.decode());driver.write_bytes(raw)
log=r/'logs'/(run+'.supervisor.log');f=log.open('xb');p=subprocess.Popen(['python3',str(driver)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(r));f.close()
print(json.dumps({'run':run,'mode':mode,'driver_pid':p.pid,'durable_dispatched':True,'completion_not_assumed':True}))
'''.replace('DRIVER',repr(base64.b64encode(driver.encode()).decode())).replace('RUN',repr(a.run)).replace('MODE',repr(a.mode))
j=json.loads(remote(script,timeout=45));out=B/'results/quick_ldm_20261026'/a.run;out.mkdir(parents=True,exist_ok=True)
(out/'launch.json').write_text(json.dumps(j,indent=2),encoding='utf-8');(out/'driver_source.py').write_text(driver,encoding='utf-8');print(json.dumps(j))
