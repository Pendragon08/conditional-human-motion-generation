from pathlib import Path
import hashlib,json,base64,datetime
root=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022')
run=root/'runs/headless_smoke_v3'
paths=[root/'logs/headless_smoke_v3_launch.json',root/'logs/headless_smoke_v3_stdout.txt',root/'logs/headless_smoke_v3_stderr.txt']
if run.exists():
 paths.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.log','.yaml','.yml'))
files={}
for p in sorted(set(paths)):
 if p.exists():
  data=p.read_bytes()
  files[str(p.relative_to(root))]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'base64':base64.b64encode(data).decode()}
proc=Path('/proc/3157748')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':3157748,'proc_exists':proc.exists(),'proc_status':(proc/'status').read_text() if proc.exists() else None,'run_exists':run.exists(),'files':files},indent=2))

