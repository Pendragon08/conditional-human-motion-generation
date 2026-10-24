"""Execute an explicit bounded command and preserve exit, logs and code hashes."""
import argparse, datetime, hashlib, json, os, pathlib, signal, subprocess, time

p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--timeout',type=int,default=10800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
if not command:raise SystemExit('missing command')
root=pathlib.Path(__file__).resolve().parents[1];os.chdir(root)
dest=root/'results'/a.name;dest.mkdir(parents=True,exist_ok=False)
hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['scripts','configs','upstream'] for f in (root/folder).rglob('*') if f.is_file() and f.suffix in ['.py','.yaml','.sh','.json'] and '__pycache__' not in str(f)}
record={'command':command,'cwd':str(root),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':a.timeout,'runner_pid':os.getpid(),'code_sha256':hashes,'status':'RUNNING'}
receipt=dest/'receipt.json';receipt.write_text(json.dumps(record,indent=2))
start=time.monotonic()
with (dest/'stdout_stderr.log').open('w') as log:
 proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record['child_pid']=proc.pid;receipt.write_text(json.dumps(record,indent=2))
 try:code=proc.wait(timeout=a.timeout)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:code=proc.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
  record['timed_out']=True
record.update(exit_code=code,elapsed_seconds=time.monotonic()-start,status='COMPLETED' if code==0 else 'FAILED',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='code_sha256'}))
raise SystemExit(0 if code==0 else 1)
