"""Update inactive candidates with prior-byte checks; never modify the active trainer."""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
from session import SSH

B = pathlib.Path(__file__).resolve().parents[1]
R = B / 'results/contact_supervision_20261026'
old = json.loads((B / 'results/filter_contact_protocol_20261026/server_cpu_v1.json').read_text(encoding='utf-8'))['deployed']
old.update(json.loads((R / 'server_cpu_author_v1.json').read_text(encoding='utf-8'))['deployed'])
names = list(old) + ['scripts/contact_filter_resources.py']
payload = {n: {'base64': base64.b64encode((B / n).read_bytes()).decode(),
               'sha256': hashlib.sha256((B / n).read_bytes()).hexdigest()} for n in names}
bound = json.loads((B / 'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
assert not set(names).intersection(bound)
local = B.parents[3] / 'textop_reproduction_20261024/filter_contact_v3_author_v4/receipt.json'
assert local.exists()
script = r'''
import ast,base64,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});payload=PAYLOAD;bound=BOUND;old=OLD
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/n)==s for n,s in bound.items())
snapshot=ROOT/'tracker_validation/contact_supervision_candidate_snapshot_v1';assert not snapshot.exists();snapshot.mkdir()
for n,row in payload.items():
 p=ROOT/n;assert not p.is_symlink() and p.parent.stat().st_uid==1013
 if p.exists():
  assert sha(p)==old[n],'Unexpected inactive candidate bytes'
  dest=snapshot/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 data=base64.b64decode(row['base64']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 if not p.exists() or sha(p)!=row['sha256']:
  temporary=p.with_name(p.name+'.contact_stage_new');assert not temporary.exists()
  with temporary.open('xb') as f:f.write(data)
  temporary.replace(p)
 assert sha(p)==row['sha256']
 if p.suffix=='.py':ast.parse(p.read_text())
spec=json.loads((ROOT/'configs/strict-contact-replay-spec.json').read_text())
assert spec['formal_evaluation_enabled'] is False and spec['real_sdk_validation_completed'] is False
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
run=ROOT/'tracker_validation/contact_supervision_integration_cpu_v1';assert not run.exists()
test=subprocess.run([sys.executable,str(ROOT/'tests/test_filter_contact_v3.py'),'--output',str(run)],env=env,capture_output=True,text=True,timeout=90)
assert test.returncode==0,(test.stdout,test.stderr)
disabled=subprocess.run([sys.executable,str(ROOT/'scripts/supervise_strict_contact_replay.py'),'--run','disabled_supervision_cpu_v2','--checkpoint','unused','--training-receipt','unused','--training-exit-receipt','unused'],env=env,capture_output=True,text=True,timeout=20)
assert disabled.returncode!=0 and 'Future exporter disabled; no SDK dispatched' in disabled.stderr
assert not (ROOT/'tracker_validation/disabled_supervision_cpu_v2').exists()
sys.path.insert(0,str(ROOT/'scripts'))
from contact_filter_resources import sample_resources
sample=sample_resources(ROOT)
assert sample['owned_group_processes']==[] and sample['owned_group_gpu_memory']==[]
try:sample_resources(ROOT,minimum_disk_bytes=10**30)
except RuntimeError as error:reserve_rejected=str(error)
else:raise AssertionError('Resource floor ignored')
assert all(sha(ROOT/n)==s for n,s in bound.items())
cpu=ROOT/'tracker_validation/contact_supervision_cpu_author_v1'
cases={}
for folder in cpu.iterdir():
 if folder.is_dir():cases[folder.name]={p.name:p.read_text() for p in folder.iterdir() if p.suffix in ['.json','.jsonl','.log']}
result={'status':'CONTACT_SUPERVISION_INACTIVE_INTEGRATION_CPU_PASSED','test_exit_code':test.returncode,'test_receipt':json.loads((run/'receipt.json').read_text()),'test_stdout':test.stdout,'test_stderr':test.stderr,'disabled_supervisor_exit_code':disabled.returncode,'disabled_supervisor_stderr':disabled.stderr,'read_only_resource_sample':sample,'resource_reserve_rejection':reserve_rejected,'deployed':{n:x['sha256'] for n,x in payload.items()},'frozen_runtime_files_verified':len(bound),'cpu_case_records':cases,'source_snapshot_preserved':str(snapshot),'new_sdk_runs':0,'physical_filter_executed':False,'dataset_accepted':False,'time_unix':time.time()}
(run/'integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''.replace('PAYLOAD', repr(payload)).replace('BOUND', repr(bound)).replace('OLD', repr(old))
transport = subprocess.run(SSH + ['python3 -'], input=script, capture_output=True,
                           text=True, encoding='utf-8', timeout=150)
proof = R / 'integration_cpu_v1.json'
assert not proof.exists()
result = json.loads(transport.stdout) if transport.stdout.strip() else {}
result.update(transport_exit_code=transport.returncode, transport_stderr=transport.stderr)
proof.write_text(json.dumps(result, indent=2), encoding='utf-8')
(R / 'candidate_runtime_bytes_v2.json').write_text(json.dumps({'files': payload}, indent=2), encoding='utf-8')
shutil.copy2(local, R / 'local_protocol_v4.json')
assert transport.returncode == 0, (transport.stderr, transport.stdout[-3000:])
print(json.dumps({k: result[k] for k in ['status', 'test_exit_code', 'disabled_supervisor_exit_code', 'frozen_runtime_files_verified', 'new_sdk_runs']}))
print('protocol_checks_passed', result['test_receipt']['checks_passed'])
