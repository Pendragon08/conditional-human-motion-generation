"""One documented production continuation; independent reviewer owns invocation."""
import base64,json,pathlib,sys
BASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'provider_tools'))
from session import remote
code=r'''import base64,hashlib,json,pathlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sp=r/'configs/teacher10000-filter-recovery-spec.json';s=json.loads(sp.read_text())
assert s['resume_enabled'] is False
source=r/'scripts/supervise_teacher10000_filter_recovery.py'
pre=subprocess.run(['taskset','-c','0-3','python3',str(source),'--preflight-only'],capture_output=True,text=True,timeout=60)
assert pre.returncode==0,pre.stderr
run=s['run'];folder=r/'teacher10000_full_filter_20261027/drivers';folder.mkdir(exist_ok=True)
driver=folder/(run+'.py');assert not driver.exists()
snapshot=folder/(run+'_disabled_spec.json');assert not snapshot.exists();snapshot.write_bytes(sp.read_bytes())
text="import pathlib,subprocess,json,time\nr=pathlib.Path("+repr(str(r))+')\n'+"started=time.time()\ncmd=['taskset','-c','0-3','python3',str(r/'scripts/supervise_teacher10000_filter_recovery.py')]\ncode=subprocess.call(cmd)\n(r/'receipts'/"+repr('filter_recovery_'+run+'_outer_exit.json')+").write_text(json.dumps({'actual_returncode':code,'elapsed_seconds':time.time()-started,'command':cmd},indent=2))\nraise SystemExit(code)\n"
compile(text,str(driver),'exec')
with driver.open('x') as f:f.write(text)
s['resume_enabled']=True;sp.write_text(json.dumps(s,indent=2))
log=r/'logs'/('filter_recovery_'+run+'_driver.log');assert not log.exists()
with log.open('xb') as f:
 child=subprocess.Popen(['taskset','-c','0-3','python3',str(driver)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=r)
result={'driver_pid':child.pid,'run':run,'active_spec_sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),'active_spec_b64':base64.b64encode(sp.read_bytes()).decode(),'preflight':json.loads(pre.stdout),'launched_once':True}
print(json.dumps(result))
'''
j=json.loads(remote(code,timeout=90));(BASE/'configs/teacher10000-filter-recovery-spec.json').write_bytes(base64.b64decode(j.pop('active_spec_b64')))
out=BASE/'results/teacher10000_full_filter_20261027/recovery_launch.json';assert not out.exists();out.write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))
