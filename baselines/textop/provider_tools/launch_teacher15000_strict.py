"""One durable frozen15k replay, with exclusive client intent and output."""
import argparse,hashlib,json,pathlib,re
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run)
expected=hashlib.sha256((B/'configs/teacher15000-direct-strict-spec.json').read_bytes()).hexdigest()
code='''import pathlib,json,hashlib,subprocess,os,fcntl
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;stage=STAGE
os.sched_setaffinity(0,{0,1,2,3});assert os.getuid()==1013
sp=r/'configs/teacher15000-direct-strict-spec.json';assert hashlib.sha256(sp.read_bytes()).hexdigest()==EXPECTED
s=json.loads(sp.read_text());assert s['pilot_launch_enabled'] and (stage=='smoke' or s['full_launch_enabled'])
prefix='teacher15000_strict_'+run
paths=[r/'tracker_validation'/run]+[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json']]
assert not any(p.exists() for p in paths),'Existing run: inspect, never relaunch'
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
cmd=['python3',str(r/'scripts/supervise_teacher15000_strict.py'),'--stage',stage,'--run',run]
driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
text="import subprocess,json,pathlib\\nq=subprocess.run("+repr(cmd)+")\\npathlib.Path("+repr(str(outer))+").write_text(json.dumps({'actual_returncode':q.returncode}))\\nraise SystemExit(q.returncode)\\n"
with driver.open('x') as f:f.write(text)
log=r/'logs'/(prefix+'_outer.log')
with log.open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=r)
result={'run':run,'stage':stage,'driver_pid':q.pid,'spec_sha256':EXPECTED,'command':cmd,'status':'DURABLE_SUPERVISOR_DISPATCHED_NOT_YET_PASSED'}
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''.replace('RUN',repr(a.run)).replace('STAGE',repr(a.stage)).replace('EXPECTED',repr(expected))
out=B/'results/teacher15000_direct_strict_20261027'/a.run;out.mkdir(parents=True,exist_ok=False)
j=json.loads(recorded_remote(code,out/'dispatch.json',timeout=60))
(out/'launch.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))
