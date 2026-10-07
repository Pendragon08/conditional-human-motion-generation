"""Exactly one interface check, no formal training or final scoring."""
import json
from r3_world_ssh_transport import ROOT,REMOTE,execute
spec=json.loads((ROOT/'configs/world_ppo_eval_sources_20261007.json').read_text())
program='''
from pathlib import Path
import subprocess,json,hashlib,shlex,fcntl
r=Path(REMOTE);prefix=r/'artifacts/monitoring/r3_world_ppo_eval_interface_20261007'
assert not prefix.with_suffix('.state').exists() and not (r/'artifacts/world_residual_ppo_20261007/eval_interface').exists()
prior=(r/'artifacts/monitoring/r3_world_ppo_capacity_20261007.state').read_text()
assert prior.startswith('COMPLETE') and 'P-history_exit=0' in prior and 'W-correct_exit=0' in prior
for name,expected in SOURCES.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
with (r/'artifacts/monitoring/r3_world_ppo_gpu0.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert int(gpu.splitlines()[0].split(',')[1])<1500,gpu
name='r3_world_ppo_eval_interface_20261007'
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
subprocess.run(['bash','-n',str(r/'scripts/r3_world_ppo_eval_interface_20261007.sh')],check=True)
subprocess.run(['bash','-n',str(r/'scripts/r3_world_ppo_formal_stage_20261007.sh')],check=True)
cmd='bash '+shlex.quote(str(r/'scripts/r3_world_ppo_eval_interface_20261007.sh'))+' > '+shlex.quote(str(prefix.with_suffix('.queue.log')))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
print('WORLD_PPO_EVAL_INTERFACE_DISPATCHED')
'''.replace('REMOTE',repr(REMOTE)).replace('SOURCES',repr(spec))
print(execute(program))
