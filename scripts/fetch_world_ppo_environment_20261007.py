"""Read only pinned task sources, never import the simulator."""
import base64,json
from r3_world_ssh_transport import execute,ROOT,REMOTE
program='''
from pathlib import Path
import subprocess,json,base64
r=Path(REMOTE)/'third_party/DreamControl/Training'
try:files=subprocess.check_output(['rg','--files',str(r/'source')],text=True).splitlines()
except FileNotFoundError:files=[str(p) for p in (r/'source').rglob('*.py')]
chosen=[f for f in files if f.endswith(('motion_tracking_kick_env.py','motion_tracking_env.py','rsl_rl_ppo_cfg.py','agent_cfg.py','motion_tracking_env_cfg.py','motion_tracking_interactive_base.py','rl_cfg.py','termination_manager.py','manager_based_rl_env.py'))]
out={}
for f in chosen:
 p=Path(f)
 if p.stat().st_size<150000:out[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(out))
'''.replace('REMOTE',repr(REMOTE))
data=json.loads(execute(program))
out=ROOT/'artifacts/monitoring/world_ppo_task_read_only'
for name,b64 in data.items():
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b64))
print(json.dumps(list(data)))
