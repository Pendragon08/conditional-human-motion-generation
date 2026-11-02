"""Read-only preflight; optional first deployment installs disabled own files only."""
import argparse
import base64
import hashlib
import json
import pathlib
import re
import subprocess
from session import SSH

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--deploy-disabled', action='store_true')
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.run)
out = B / 'results/quick_reproduction_20261026'
out.mkdir(exist_ok=True)
proof = out / (a.run + '.json')
assert not proof.exists(), 'Preserve earlier witness results'
names = ['scripts/quick_budget_guards.py', 'scripts/quick_budget_control.py', 'scripts/quick_pidfd.py',
         'configs/quick-reproduction-plan-20261026.json']
payload = {n: {'base64':base64.b64encode((B/n).read_bytes()).decode(),
               'sha256':hashlib.sha256((B/n).read_bytes()).hexdigest()} for n in names}
bound = json.loads((B/'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
assert not set(names).intersection(bound)
script = r'''
import ast,base64,hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});payload=PAYLOAD;bound=BOUND;deploy=DEPLOY
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/n)==v for n,v in bound.items())
for n,row in payload.items():
 path=root/n
 if not path.exists():
  assert deploy and not path.is_symlink() and path.parent.stat().st_uid==1013
  with path.open('xb') as f:f.write(base64.b64decode(row['base64']))
 assert sha(path)==row['sha256']
 if path.suffix=='.py':ast.parse(path.read_text())
plan=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text());assert plan['early_stop_enabled'] is False
result=subprocess.run([sys.executable,str(root/'scripts/quick_budget_control.py'),'--action','observe'],capture_output=True,text=True,timeout=30)
assert result.returncode==0,(result.stdout,result.stderr)
observation=json.loads(result.stdout);assert observation['signals_sent']==[] and observation['sdk_runs']==0 and observation['enabled'] is False
assert all(sha(root/n)==v for n,v in bound.items())
print(json.dumps({'status':'QUICK_BUDGET_DISABLED_PREFLIGHT_VERIFIED','observation':observation,'frozen_runtime_files_verified':len(bound),'deployed_files':{n:x['sha256'] for n,x in payload.items()},'signals_sent':0,'new_sdk_runs':0,'stdout':result.stdout,'stderr':result.stderr}))
'''.replace('PAYLOAD',repr(payload)).replace('BOUND',repr(bound)).replace('DEPLOY',repr(a.deploy_disabled))
transport = subprocess.run(SSH+['python3 -'], input=script, capture_output=True,
                           text=True, encoding='utf-8', timeout=60)
r = json.loads(transport.stdout) if transport.stdout.strip() else {}
r.update(transport_exit_code=transport.returncode, transport_stderr=transport.stderr)
proof.write_text(json.dumps(r,indent=2),encoding='utf-8')
assert transport.returncode==0,(transport.stderr,transport.stdout[-3000:])
print(json.dumps({'status':r['status'],'completed':r['observation'].get('current_completed_iterations'),
                  'checkpoint_ready':r['observation']['target_checkpoint_ready'],'signals_sent':0}))
