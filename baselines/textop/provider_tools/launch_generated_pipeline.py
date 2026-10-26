"""Durable exactly-once wrapper for the two-stage generated-ref SDK trial."""
import argparse,ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isidentifier() and a.run.isascii()
driver="""import subprocess,pathlib,json,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
z=subprocess.run(['python3',str(r/'scripts/supervise_generated_pipeline.py'),'--run',RUN],capture_output=True,text=True,timeout=1950)
(r/'receipts'/(RUN+'_supervisor_execution.json')).write_text(json.dumps(dict(exit_code=z.returncode,stdout=z.stdout,stderr=z.stderr,time_unix=time.time()),indent=2))
""".replace('RUN',repr(a.run));ast.parse(driver)
script=r'''
import ast,base64,fcntl,json,pathlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;assert not(r/'generated_execution_private'/run).exists()
assert json.loads((r/'configs/quick-generated-sdk-spec.json').read_text())['validation_enabled']
lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lk,fcntl.LOCK_UN)
p=r/'receipts'/(run+'_driver.py');assert not p.exists();raw=base64.b64decode(DRIVER);ast.parse(raw.decode());p.write_bytes(raw)
f=(r/'logs'/(run+'.supervisor.log')).open('xb');child=subprocess.Popen(['python3',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(r));f.close();print(json.dumps({'run':run,'driver_pid':child.pid,'durable_dispatch_only':True,'completion_not_assumed':True}))
'''.replace('DRIVER',repr(base64.b64encode(driver.encode()).decode())).replace('RUN',repr(a.run))
j=json.loads(remote(script,timeout=45));out=B/'results/generated_execution_20261026'/a.run;out.mkdir(parents=True,exist_ok=True);(out/'launch.json').write_text(json.dumps(j,indent=2),encoding='utf-8');(out/'driver_source.py').write_text(driver,encoding='utf-8');print(json.dumps(j))
