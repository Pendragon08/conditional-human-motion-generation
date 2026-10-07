"""Start the single locked 15-run queue once after actual independent gates."""
import json
from r3_world_ssh_transport import ROOT,REMOTE,execute
sources=json.loads((ROOT/'configs/world_ppo_formal_sources_20261007.json').read_text())
program='''
from pathlib import Path
import subprocess,json,hashlib,shlex,fcntl,time
r=Path(REMOTE);prefix=r/'artifacts/monitoring/r3_world_ppo_formal_20261007'
name='r3_world_ppo_formal_20261007'
assert not prefix.with_suffix('.state').exists() and not (r/'artifacts/world_residual_ppo_20261007/formal').exists()
assert not prefix.with_suffix('.dispatch.json').exists(),'Prior unknown dispatch must be audited, never duplicated'
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
gate=json.loads((r/'artifacts/monitoring/r3_world_residual_ppo_local_view_gate_20261007.json').read_text())
assert gate['user_confirmed_local_stream'] and gate['source']=='actual_isaaclab_simulation'
assert gate['user_answer']=='能看到机器人持续运动' and gate['local_url'].endswith('/live')
for required in ['runtime','capacity','eval_interface']:
 receipt=json.loads((r/f'artifacts/monitoring/r3_world_ppo_{required}_independent_ready_20261007.json').read_text())
 assert receipt['ready_for_formal']==True and receipt['formal_PPO_launched']==False,required
for seed in [42,43,44]:
 assert json.loads((r/f'artifacts/world_confirmation_20261007/confirmation_v2/evaluate_seed{seed}_v1/summary.json').read_text())['formal_prediction_gate_passed']
for filename,expected in SOURCES.items():assert hashlib.sha256((r/filename).read_bytes()).hexdigest()==expected,filename
for seed in [42,43,44]:
 for arm in ['P0','P-cap','P-history','W-correct','W-shuffled-original']:
  for mode in ['train','eval']:
   assert not (r/f'artifacts/monitoring/r3_world_ppo_{mode}_{arm}_{seed}_20261007.state').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert int(gpu.splitlines()[0].split(',')[1])<1500,gpu
disk=__import__('shutil').disk_usage(str(r));assert disk.free>15*1024**3
with (r/'artifacts/monitoring/r3_world_ppo_gpu0.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
subprocess.run(['bash','-n',str(r/'scripts/r3_world_ppo_formal_stage_v3_20261007.sh')],check=True)
launcher={'dispatch_unix':time.time(),'GPU':0,'independent_viewer_GPU':1,'gate_sha256':hashlib.sha256((r/'artifacts/monitoring/r3_world_residual_ppo_local_view_gate_20261007.json').read_bytes()).hexdigest(),'sources':SOURCES,
 'arms':5,'seeds':[42,43,44],'updates_per_arm_seed':500,'num_envs':8192,
 'training_transitions_planned':1474560000,'evaluation_transitions_planned':15360000,
 'PPO_formal_progress_before_dispatch':0,'additional_IK':False}
(r/'artifacts/monitoring/r3_world_ppo_formal_20261007.dispatch.json').write_text(json.dumps(launcher,indent=2)+'\\n')
cmd='cd '+shlex.quote(str(r))+' && /usr/bin/python3 -u scripts/r3_world_ppo_formal_queue_20261007.py > '+shlex.quote(str(prefix.with_suffix('.queue.log')))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
print('WORLD_PPO_FORMAL_DISPATCHED_ONCE',json.dumps(launcher))
'''.replace('REMOTE',repr(REMOTE)).replace('SOURCES',repr(sources))
print(execute(program))
