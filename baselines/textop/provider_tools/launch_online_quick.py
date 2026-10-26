"""One durable declared online invocation, unique run/exit paths; never retries."""
import argparse,json,pathlib
from session import remote
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isidentifier()
script=r'''
import json,os,pathlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN
assert not (r/'online_private'/run).exists();outer=r/'receipts'/('online_'+run+'_outer_exit.json');assert not outer.exists()
driver=r/'receipts'/('online_'+run+'_driver.py');assert not driver.exists()
code="import subprocess,json,pathlib\nr=pathlib.Path(%r)\np=subprocess.run(['python3','scripts/supervise_online_quick.py','--run',%r],cwd=r,timeout=950)\n(r/'receipts'/%r).write_text(json.dumps({'actual_returncode':p.returncode}))\n"%(str(r),run,outer.name)
driver.write_text(code)
with (r/'logs'/('online_'+run+'_outer.log')).open('xb') as f:
 p=subprocess.Popen(['python3',str(driver)],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'driver_pid':p.pid,'run':run,'driver':str(driver),'outer_exit':str(outer)}))
'''.replace('RUN',repr(a.run))
print(remote(script,timeout=60))
