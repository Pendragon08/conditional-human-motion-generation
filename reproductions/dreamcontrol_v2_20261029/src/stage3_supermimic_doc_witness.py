"""Execute the documented bounded SuperMimic/PhysX integration witness once."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

D=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103')
spec=json.loads((D/'runtime_bridge_spec.json').read_text())
digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
assert digest=='c7724ac5'
for name,value in spec['source_hashes'].items():
    assert hashlib.sha256((D/'src'/name).read_bytes()).hexdigest()==value
launch=D/'doc_witness_launch_v1';launch.mkdir(exist_ok=False)
output=D/'reports/physx_model_doc_witness_v1';assert not output.exists()
command=[spec['python'],'-u',str(D/'src/stage3_supermimic_pick_smoke.py'),
    '--headless','--device','cuda:1','--num-envs','4','--steps','64','--model-updates','16',
    '--output-dir',str(output)]
(launch/'intent.json').open('x').write(json.dumps({'command':command,'spec_hash':digest,
    'scope':'independent documented invocation; real4-environment integration, not formal training'},indent=2))
with (launch/'child.log').open('x') as log:
    child=subprocess.Popen(['timeout','--kill-after=20s','180']+command,env={**os.environ,**spec['env']},
        cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (launch/'child.json').open('x').write(json.dumps({'pid':child.pid}))
    code=child.wait()
(launch/'exit.json').open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
if code!=0:
    print('WITNESS_FAILED inspect '+str(launch/'child.log'));sys.exit(code if code>0 else 1)
result=json.loads((output/'status.json').read_text())
assert result['status']=='integration_smoke_completed'
assert result['real_vector_steps']==64 and result['world_model_updates']==16 and result['policy_updates']==1
assert result['live_observation_max_error']<1e-4 and all(result['checkpoint_roundtrip'].values())
print(json.dumps({'sentinel':'SUPERMIMIC_PHYSX_DOC_WITNESS_PASSED','returncode':code,
    'spec_hash':digest,'result_path':str(output/'status.json'),
    'result_sha256':hashlib.sha256((output/'status.json').read_bytes()).hexdigest(),
    'actual_environment_transitions':result['actual_environment_transitions'],
    'policy_parameter_change':result['maximum_policy_parameter_change'],
    'world_model_updates':result['world_model_updates'],'task_success_claimed':False},indent=2))
