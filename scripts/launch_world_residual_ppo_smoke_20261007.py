"""Launch once only: five diagnostic arms, then bounded actual-scene preflight."""
from r3_world_ssh_transport import execute

PROGRAM=r'''
from pathlib import Path
import json,subprocess,hashlib,fcntl,time,shlex
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
name='r3_world_residual_ppo_smoke_20261007'
m=r/'artifacts/monitoring';out=r/'artifacts/world_residual_ppo_20261007'
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
assert not (m/(name+'.state')).exists()
assert all(not (out/('smoke_'+arm+'_seed42_v1')).exists() for arm in ['P0','P-cap','P-history','W-correct','W-shuffled-original'])
assert not (out/'visibility_seed42_v1').exists()
for seed in [42,43,44]:
 data=json.loads((r/f'artifacts/world_confirmation_20261007/confirmation_v2/evaluate_seed{seed}_v1/summary.json').read_text())
 assert data['formal_prediction_gate_passed'] and data['PPO_updates']==0
files=json.loads((r/'configs/world_residual_ppo_runtime_sources_20261007.json').read_text())
assert all(hashlib.sha256((r/name).read_bytes()).hexdigest()==h for name,h in files.items())
for name in ['scripts/r3_world_residual_ppo_smoke_20261007.sh','scripts/r3_world_residual_visibility_20261007.sh']:
 assert b'\r' not in (r/name).read_bytes()
 subprocess.run(['bash','-n',str(r/name)],check=True)
with (m/'r3_world_ppo_gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
mem=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
assert mem<1500,('GPU0 active/busy',mem)
assert subprocess.check_output(['df','--output=avail','-B1',str(r)],text=True).splitlines()[-1].strip().isdigit()
assert int(subprocess.check_output(['df','--output=avail','-B1',str(r)],text=True).splitlines()[-1])>15*1024**3
queue=r/'scripts/r3_world_residual_ppo_diagnostic_queue_20261007.py'
cmd='cd '+shlex.quote(str(r))+' && /usr/bin/python3 -u '+shlex.quote(str(queue))+' > '+shlex.quote(str(m/'r3_world_residual_ppo_smoke_20261007.log'))+' 2>&1'
assert '\r' not in cmd
subprocess.run(['tmux','new-session','-d','-s','r3_world_residual_ppo_smoke_20261007',cmd],check=True)
print('WORLD_RESIDUAL_PPO_DIAGNOSTIC_DISPATCHED',flush=True)
'''
if __name__=='__main__':print(execute(PROGRAM),end='')
