"""Dispatch exactly once, after the prior visibility/audit releases GPU0."""
import hashlib,json
from r3_world_ssh_transport import ROOT,REMOTE,execute
spec=json.loads((ROOT/'configs/world_ppo_capacity_sources_20261007.json').read_text())
program='''
from pathlib import Path
import subprocess,json,hashlib,shlex,fcntl
r=Path(REMOTE)
state=r/'artifacts/monitoring/r3_world_ppo_capacity_20261007.state'
session='r3_world_ppo_capacity_20261007'
assert not state.exists()
assert subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
assert json.loads((r/'artifacts/monitoring/r3_world_residual_ppo_local_view_gate_20261007.json').read_text())['user_confirmed_local_stream']
prior=json.loads((r/'artifacts/monitoring/r3_world_residual_ppo_smoke_20261007.state').read_text())
assert prior['status']=='COMPLETE' and prior['visibility_wrapper_exit']==0
for name,expected in SOURCES.items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert int(gpus.splitlines()[0].split(',')[1])<1500,gpus
with (r/'artifacts/monitoring/r3_world_ppo_gpu0.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
subprocess.run(['bash','-n',str(r/'scripts/r3_world_ppo_capacity_20261007.sh')],check=True)
cmd='bash '+shlex.quote(str(r/'scripts/r3_world_ppo_capacity_20261007.sh'))+' > '+shlex.quote(str(r/'artifacts/monitoring/r3_world_ppo_capacity_20261007.queue.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
print('WORLD_PPO_CAPACITY_DISPATCHED')
'''.replace('REMOTE',repr(REMOTE)).replace('SOURCES',repr(spec))
print(execute(program))
