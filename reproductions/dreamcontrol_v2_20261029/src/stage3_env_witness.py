"""Bounded exact-invocation witness for the existing Isaac environment; no PPO."""
import json
from pathlib import Path
import os
import signal
import subprocess

D=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
spec=json.loads((D/'env_spec_v2.json').read_text())
command=spec['smoke_command']
output=Path(command[-1])
assert not output.exists(), 'witness already exists: inspect it; never overwrite/relaunch blindly'
with (D/'reports/env_doc_witness.log').open('x') as log:
    process=subprocess.Popen(['timeout','90']+command,env={**os.environ,**spec['env']},
        cwd=spec['cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=100)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        raise
if code:
    print((D/'reports/env_doc_witness.log').read_text()[-5000:])
    raise SystemExit(code)
result=json.loads((output/'status.json').read_text())
assert result['status']=='smoke_completed' and result['policy_updates']==0
assert result['action_dimension']==31 and result['actor_observation_dimension']==102
assert result['geometry_matches_pinned_urdf'] and result['reference_mutation_actor_difference']==0
print('WITNESS '+json.dumps({'status':result['status'],'actions':31,'actor_obs':102,
    'seed':20261103,'physical_body_joints':29,'policy_updates':0,'source':'real IsaacLab/PhysX',
    'geometry_max_error_m':max(result['geometry_calibration_max_position_errors_m'])}),flush=True)
