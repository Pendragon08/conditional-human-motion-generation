from pathlib import Path
import hashlib,json,subprocess,datetime
root=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022')
code=root/'code_v9'
parent=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
files=sorted(p for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_exists':(root/'runs/headless_smoke_v3').exists(),'source_manifest':{str(p.relative_to(code)):{'sha256':sha(p),'size':p.stat().st_size} for p in files},'parent':{'path':str(parent),'sha256':sha(parent),'size':parent.stat().st_size},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],universal_newlines=True)},indent=2))
