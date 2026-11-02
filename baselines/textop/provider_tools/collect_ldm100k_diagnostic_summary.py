"""Read summary status/closure without loading models or arrays."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.run.isidentifier() and a.run.isascii() and a.evidence.isidentifier() and a.evidence.isascii()
out = B / 'results/ldm100k_user_handoff_20261028' / a.run
assert out.is_dir()
code = r'''
import json,pathlib,sys,os,fcntl,shutil,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned
run=RUN;prefix='ldm100k_diagnostic_summary_'+run
req=read(r/'receipts'/(prefix+'_request.json'))
parent=pathlib.Path(req.get('output_root',str(r/'ldm100k_user_handoff_20261028')))
files={};hashes={}
for key,p in dict(receipt=parent/run/'receipt.json',request=r/'receipts'/(prefix+'_request.json'),**{k:r/'receipts'/(prefix+v) for k,v in [('process','_process_exit.json'),('exit','_exit.json'),('outer','_outer_exit.json')]}).items():
 if p.is_file():files[key]=read(p);hashes[key]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['summarize_ldm100k_diagnostic.py','supervise_ldm100k_diagnostic_summary.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],argv=argv,state=st[0]))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for suffix in ['.log','_outer.log']:
 p=r/'logs'/(prefix+suffix)
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3500));logs[suffix]=f.read().decode(errors='replace')
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_exit_code']==0 and not workers and free
if closed:
 assert files['receipt']['output_writers_closed']
 assert files['receipt']['source_sha256']==req['executor_sha256'] and files['exit']['source_sha256']==req['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and hashes['request']==files['exit']['request_sha256']
 assert sha(r/'logs'/(prefix+'_resources.jsonl'))==files['exit']['resource_sha256']
 for item in files['receipt']['private_artifacts'].values():
  p=owned(item['path'],pathlib.Path(req.get('allowed_root',str(r))))
  assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),logs=logs,disks={str(p):shutil.disk_usage(p).free for p in [r,pathlib.Path('/home/ouyanghao')]})))
'''.replace('RUN', repr(a.run))
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=120))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
