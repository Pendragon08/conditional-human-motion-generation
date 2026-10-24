from pathlib import Path
import base64, datetime, hashlib, json, os
root=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022')
run=root/'runs/headless_smoke_v2'
files={'receipt.json':run/'receipt.json','executed_sources.json':run/'executed_sources.json','headless_smoke_v2_stdout.txt':root/'logs/headless_smoke_v2_stdout.txt','headless_smoke_v2_stderr.txt':root/'logs/headless_smoke_v2_stderr.txt','headless_smoke_v2_launch.json':root/'logs/headless_smoke_v2_launch.json'}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':3156839,'process_present':Path('/proc/3156839').exists(),'files':{}}
for name,path in files.items():
 if path.is_file():
  data=path.read_bytes()
  result['files'][name]={'path':str(path),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode('ascii')}
 else: result['files'][name]={'path':str(path),'exists':False}
if Path('/proc/3156839/status').is_file(): result['process_status']=Path('/proc/3156839/status').read_text()
print(json.dumps(result))
