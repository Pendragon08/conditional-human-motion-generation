"""Stop only the proven failed diagnostic, retaining its actual exit evidence."""
from r3_world_ssh_transport import execute, REMOTE
program=r'''
from pathlib import Path
import os,signal,json,time,hashlib,shutil
r=Path(REMOTE);pid=1685179;p=Path('/proc')/str(pid)
command=p.joinpath('cmdline').read_bytes().replace(b'\x00',b' ').decode()
assert 'r3_world_ppo_eval_isaac_20261007.py' in command and '--mode diagnostic' in command
assert 'eval-diagnostic_P-history_seed42_v1' in command and '--num_envs 64' in command
assert not (r/'artifacts/monitoring/r3_world_ppo_formal_20261007.state').exists()
prefix=r/'artifacts/monitoring/r3_world_ppo_eval-diagnostic_P-history_42_20261007'
log=prefix.with_suffix('.log').read_bytes()
assert b'WORLD_PPO_EVAL_STEP' not in log and b'WORLD_PPO_CONTROL_EVALUATION_COMPLETE' not in log
assert log.count(b'Timeout (0:04:00)')>=2 and b'simulation_app.py' in log and b'in close' in log
out=r/'artifacts/world_residual_ppo_20261007/eval_interface/eval-diagnostic_P-history_seed42_v1'
assert out.exists() and not list(out.iterdir())
backup=r/'artifacts/recovery/world_ppo_eval_interface_hang_20261007_v1'
assert not backup.exists();backup.mkdir(parents=True)
for suffix in ['.state','.log']:shutil.copy2(prefix.with_suffix(suffix),backup/('original'+suffix))
stat=p.joinpath('stat').read_text()
receipt={'diagnostic_pid':pid,'command':command,'proc_stat':stat,'observed_unix':time.time(),
 'original_log_sha256':hashlib.sha256(log).hexdigest(),'original_log_bytes':len(log),
 'verified_output_files':[],'formal_training_exists':False,'source_unchanged':True,
 'reason':'Main thread already in app.close, repeated >=8min teardown dump; no control trace or step sentinel',
 'action':'SIGTERM exact diagnostic child only; not natural timeout; wrapper must record actual rc'}
(backup/'parent_termination.json').write_text(json.dumps(receipt,indent=2)+'\n')
os.kill(pid,signal.SIGTERM)
print(json.dumps(receipt))
'''.replace('REMOTE',repr(REMOTE))
print(execute(program))
