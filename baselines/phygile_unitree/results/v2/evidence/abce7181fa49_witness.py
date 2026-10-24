from pathlib import Path
import base64, datetime, hashlib, importlib.util, json, re, subprocess, sys, time

W = Path(r'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/headless_smoke_doc_witness_v2')
B = Path(r'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree')
PS = r'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
CMD = "& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/submit_isaac_job.py' --version code_v8 --name headless_smoke_v2 --mode train --num-envs 64 --iterations 2"

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save_json(name, value): (W / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def sha(data): return hashlib.sha256(data).hexdigest()

inputs = {
 'run-experiment_SKILL.md': Path(r'C:/Users/Pende/.codex/skills/run-experiment/SKILL.md'),
 'compute-env-contract.md': Path(r'C:/Users/Pende/.codex/skills/shared-references/compute-env-contract.md'),
 'COMPUTE_LEDGER.md': B/'docs/COMPUTE_LEDGER.md',
 'HEADLESS_V2_INVOCATION.md': B/'docs/HEADLESS_V2_INVOCATION.md',
 'headless_authorization_20261022.json': B/'configs/headless_authorization_20261022.json',
 'submit_isaac_job.py': B/'scripts/submit_isaac_job.py',
 'environment_plan_v2.json': B/'configs/environment_plan_v2.json',
}
snapshots = W/'document_snapshots'
snapshots.mkdir(exist_ok=False)
input_hashes = {}
for name, path in inputs.items():
 data = path.read_bytes()
 (snapshots/name).write_bytes(data)
 input_hashes[str(path)] = sha(data)
save_json('document_hashes.json', input_hashes)
doc_cmd = re.search(r'```powershell\s*\n(.*?)\n```', (B/'docs/HEADLESS_V2_INVOCATION.md').read_text(encoding='utf-8'), re.S).group(1).strip()
assert CMD == doc_cmd, 'Expected command differs from documented invocation'
(W/'command.txt').write_text(CMD+'\n', encoding='utf-8')
with (W/'invocation_started.json').open('x', encoding='utf-8') as f:
 json.dump({'utc':now(),'command':CMD,'shell':PS,'exact_document_match':True,'attempt':1}, f, ensure_ascii=False, indent=2)
started = time.monotonic()
r = subprocess.run([PS,'-NoLogo','-NoProfile','-NonInteractive','-Command',CMD], capture_output=True)
(W/'submit_stdout.txt').write_bytes(r.stdout)
(W/'submit_stderr.txt').write_bytes(r.stderr)
stdout = r.stdout.decode('utf-8', errors='replace')
submission = {'utc':now(),'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-started,'command':CMD,'submit_stdout_sha256':sha(r.stdout),'submit_stderr_sha256':sha(r.stderr),'launch':None}
for line in stdout.splitlines():
 try:
  val=json.loads(line)
  if isinstance(val,dict) and 'pid' in val: submission['launch']=val
 except (ValueError,TypeError): pass
save_json('submission.json', submission)
print(json.dumps({'phase':'submitted','exit_code':r.returncode,'launch':submission['launch']},ensure_ascii=True), flush=True)
if r.returncode != 0 or not submission['launch']:
 save_json('acceptance_report.json',{'utc':now(),'accepted':False,'submission':submission,'divergences':['Documented invocation did not return a successful launch record.'],'repairs_or_retries':0})
 sys.exit(r.returncode or 2)

spec = importlib.util.spec_from_file_location('task_ops', r'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/ops.py')
ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
pid = submission['launch']['pid']
remote_script = r'''from pathlib import Path
import base64, datetime, hashlib, json, os
root=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022')
run=root/'runs/headless_smoke_v2'
files={'receipt.json':run/'receipt.json','executed_sources.json':run/'executed_sources.json','headless_smoke_v2_stdout.txt':root/'logs/headless_smoke_v2_stdout.txt','headless_smoke_v2_stderr.txt':root/'logs/headless_smoke_v2_stderr.txt','headless_smoke_v2_launch.json':root/'logs/headless_smoke_v2_launch.json'}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':PID,'process_present':Path('/proc/PID').exists(),'files':{}}
for name,path in files.items():
 if path.is_file():
  data=path.read_bytes()
  result['files'][name]={'path':str(path),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode('ascii')}
 else: result['files'][name]={'path':str(path),'exists':False}
if Path('/proc/PID/status').is_file(): result['process_status']=Path('/proc/PID/status').read_text()
print(json.dumps(result))
'''.replace('PID', str(pid))
(W/'read_only_poll.py').write_text(remote_script,encoding='utf-8')
poll_start = time.monotonic()
final = None
for i in range(21):
 if i: time.sleep(30)
 if time.monotonic()-poll_start >= 600: break
 poll_started=time.monotonic()
 try:
  response=ops.remote(remote_script,timeout=30)
  record={'index':i,'local_utc':now(),'exit_code':response.returncode,'stdout':response.stdout,'stderr':response.stderr,'elapsed_seconds':time.monotonic()-poll_started}
  save_json('poll_%02d_transport.json'%i,record)
  if response.returncode: 
   print(json.dumps({'phase':'poll_error','index':i,'exit_code':response.returncode}),flush=True)
   continue
  current=json.loads(response.stdout)
  save_json('latest_status.json',{k:v for k,v in current.items() if k!='files'})
  remote_files=W/'remote_files'; remote_files.mkdir(exist_ok=True)
  for name, details in current['files'].items():
   if 'base64' in details:
    data=base64.b64decode(details['base64']); assert sha(data)==details['sha256']
    (remote_files/name).write_bytes(data)
  final=current
  receipt_entry=current['files']['receipt.json']
  receipt=json.loads(base64.b64decode(receipt_entry['base64'])) if 'base64' in receipt_entry else None
  print(json.dumps({'phase':'poll','index':i,'process_present':current['process_present'],'receipt_status':receipt.get('status') if receipt else None,'stdout_size':current['files']['headless_smoke_v2_stdout.txt'].get('size'),'stderr_size':current['files']['headless_smoke_v2_stderr.txt'].get('size')},ensure_ascii=True),flush=True)
  if receipt is not None or not current['process_present']: break
 except Exception as exc:
  save_json('poll_%02d_exception.json'%i,{'utc':now(),'type':type(exc).__name__,'message':str(exc)})
  print(json.dumps({'phase':'poll_exception','index':i,'type':type(exc).__name__,'message':str(exc)},ensure_ascii=True),flush=True)

report={'utc':now(),'accepted':False,'submission':submission,'repairs_or_retries':0,'poll_count':i+1,'checks':{},'divergences':[]}
if final is None:
 report['divergences'].append('No read-only remote status was returned successfully within the bounded observation period.')
else:
 receipt_entry=final['files']['receipt.json']
 if 'base64' not in receipt_entry:
  report['divergences'].append('No receipt.json was present when the process exited or the 10-minute observation limit was reached.')
 else:
  receipt=json.loads(base64.b64decode(receipt_entry['base64']))
  report['receipt']=receipt
  expected={'status':'BOUNDED_TRAINING_COMPLETED','headless':True,'optimizer_loaded_exactly':True,'additional_ppo_iterations':2,'actual_adam_calls':40,'executed_sources_unchanged':True,'parent_unchanged':True}
  for key,value in expected.items(): report['checks'][key]={'expected':value,'actual':receipt.get(key),'passed':receipt.get(key)==value}
  report['checks']['changed_policy_tensors']={'expected':'> 0','actual':receipt.get('changed_policy_tensors'),'passed':isinstance(receipt.get('changed_policy_tensors'),(int,float)) and receipt.get('changed_policy_tensors')>0}
  executed=final['files']['executed_sources.json']
  report['executed_sources_sha256_actual']=executed.get('sha256')
  report['receipt_binding_fields']={k:v for k,v in receipt.items() if 'source' in k or 'sha' in k or 'hash' in k}
  for key,check in report['checks'].items():
   if not check['passed']: report['divergences'].append('%s expected %r; actual %r'%(key,check['expected'],check['actual']))
 report['remote_file_hashes']={name:{k:v for k,v in info.items() if k!='base64'} for name,info in final['files'].items()}
report['accepted']=not report['divergences'] and bool(report['checks'])
save_json('acceptance_report.json',report)
print(json.dumps({'phase':'observation_complete','provisional_acceptance':report['accepted'],'divergences':report['divergences'],'output':str(W)},ensure_ascii=True),flush=True)
