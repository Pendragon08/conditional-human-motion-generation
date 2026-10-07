"""Read-only fixed upstream sources for metric/interface audit."""
from pathlib import Path
import base64
import json
from r3_world_ssh_transport import execute,ROOT
PROGRAM=r'''
from pathlib import Path
import json,base64
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
out={}
for p in r.rglob('*.py'):
 if any(x in p.parts for x in ['logs','.git']):continue
 if p.name in ['motion_lib_base.py','play_eval.py','g1_env_cfg.py'] or (p.name=='rewards.py' and 'motion_tracking' in str(p)):
  if p.stat().st_size<150000:out[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(out))
'''
if __name__=='__main__':
    out=ROOT/'artifacts/monitoring/world_ppo_task_read_only'
    data=json.loads(execute(PROGRAM))
    for name,payload in data.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(payload))
    print('READ_ONLY_SOURCES',list(data))
