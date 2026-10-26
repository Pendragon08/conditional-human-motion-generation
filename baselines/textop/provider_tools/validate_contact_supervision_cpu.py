"""Stage two new files and test owned CPU children; no SDK tasks are dispatched."""
import base64
import hashlib
import json
import pathlib
import subprocess
from session import SSH

B = pathlib.Path(__file__).resolve().parents[1]
R = B / 'results/contact_supervision_20261026'
R.mkdir(exist_ok=True)
names = ['scripts/contact_process_supervision.py', 'tests/test_contact_process_supervision.py']
payload = {n: {'base64': base64.b64encode((B / n).read_bytes()).decode(),
               'sha256': hashlib.sha256((B / n).read_bytes()).hexdigest()} for n in names}
bound = json.loads((B / 'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
assert not set(names).intersection(bound)
script = r'''
import ast,base64,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
payload=PAYLOAD;bound=BOUND
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/n)==s for n,s in bound.items()),'Frozen formal runtime differs'
for n,row in payload.items():
 p=ROOT/n;assert not p.is_symlink() and p.parent.stat().st_uid==1013
 if p.exists():assert sha(p)==row['sha256'],'Never replace a different candidate'
 else:
  data=base64.b64decode(row['base64']);assert hashlib.sha256(data).hexdigest()==row['sha256']
  with p.open('xb') as f:f.write(data)
 ast.parse(p.read_text());assert sha(p)==row['sha256']
run=ROOT/'tracker_validation/contact_supervision_cpu_author_v1';assert not run.exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
result=subprocess.run([sys.executable,str(ROOT/'tests/test_contact_process_supervision.py'),'--output',str(run)],env=env,capture_output=True,text=True,timeout=60)
assert all(sha(ROOT/n)==s for n,s in bound.items()),'Frozen formal runtime changed'
receipt={'test_exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'python':sys.version,'deployed':{n:x['sha256'] for n,x in payload.items()},'frozen_runtime_files_verified':len(bound),'new_sdk_runs':0,'time_unix':time.time()}
if (run/'receipt.json').exists():receipt['test_receipt']=json.loads((run/'receipt.json').read_text())
(run/'execution.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
raise SystemExit(result.returncode)
'''.replace('PAYLOAD', repr(payload)).replace('BOUND', repr(bound))
transport = subprocess.run(SSH + ['python3 -'], input=script, capture_output=True,
                           text=True, encoding='utf-8', timeout=100)
proof = R / 'server_cpu_author_v1.json'
assert not proof.exists()
result = json.loads(transport.stdout) if transport.stdout.strip() else {}
result['transport_exit_code'] = transport.returncode
result['transport_stderr'] = transport.stderr
proof.write_text(json.dumps(result, indent=2), encoding='utf-8')
(R / 'tested_runtime_bytes.json').write_text(json.dumps({'files': payload}, indent=2), encoding='utf-8')
assert transport.returncode == 0, (transport.stderr, transport.stdout[-3000:])
print(json.dumps({k: result[k] for k in ['test_exit_code', 'frozen_runtime_files_verified', 'new_sdk_runs']}))
print(json.dumps(result['test_receipt']))
