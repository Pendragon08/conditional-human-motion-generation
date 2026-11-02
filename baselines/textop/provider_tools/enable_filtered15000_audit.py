"""Enable CPU-only full result audit once real full replay closes; no dispatch."""
import pathlib,json,hashlib,base64
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
sp=B/'configs/teacher15000-full-result-audit-spec.json';old=sp.read_bytes();s=json.loads(old)
assert s['launch_enabled'] is False
proof=B/'results/filtered15000_handoff_tools_20261027/cpu_static_v1.json';j=json.loads(proof.read_text(encoding='utf-8'))
assert j['status']=='NEW_HANDOFF_CPU_PRIMITIVE_AND_STATIC_CHECKS_PASSED' and j['new_specs_all_disabled']
assert j['source_sha256']['scripts/audit_teacher15000_full_result.py']==s['executor_sha256']
s['launch_enabled']=True;s['cpu_primitive_check_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest();new=(json.dumps(s,indent=2)+'\n').encode()
expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,os,sys,fcntl,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/teacher15000-full-result-audit-spec.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==OLD
data=base64.b64decode(NEW);s=json.loads(data);sys.path.insert(0,str(r/'scripts'))
from audit_teacher15000_full_result import load_closed_run
load_closed_run()
for name,h in dict(s['bound_sources'],**{'scripts/audit_teacher15000_full_result.py':s['executor_sha256'],'scripts/supervise_teacher15000_full_audit.py':s['supervisor_sha256']}).items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(r).free>=s['minimum_disk_free_bytes']
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0];assert mem>=s['minimum_ram_free_bytes']
archive=r/'receipts/filtered15000_cpu_audit_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes())
temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(p)
proof=dict(status='CLOSED_FULL_INPUT_CPU_AUDIT_ENABLED_NOT_DISPATCHED',old_spec_sha256=OLD,new_spec_sha256=hashlib.sha256(data).hexdigest(),
 full_execution_closed=True,free_disk_bytes=shutil.disk_usage(r).free,free_ram_bytes=mem,new_gpu_contexts=0,new_model_loads=0,new_cpu_audits=0)
with (r/'receipts/filtered15000_cpu_audit_activation.json').open('x') as f:json.dump(proof,f,indent=2)
lock.close();print(json.dumps(proof))
'''.replace('OLD',repr(expected)).replace('NEW',repr(base64.b64encode(new).decode()))
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/'enable_audit_v1_client.json',timeout=90));sp.write_bytes(new)
(folder/'audit_activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
