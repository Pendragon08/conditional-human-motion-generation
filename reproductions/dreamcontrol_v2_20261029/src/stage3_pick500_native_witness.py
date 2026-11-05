"""Bounded native replay and real reset witness for the four actual new candidates."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
spec=json.loads((R/'stage3_supermimic29_20261103/runtime_bridge_spec.json').read_text())
launch=O/'native_pilot_v1_launch';launch.mkdir(exist_ok=False)
post=O/'postopt_pilot_jit_v1'
report=json.loads((post/'report.json').read_text());assert report['completed']==4 and report['kinematic_passed']==4
candidates=[post/('candidate_'+str(i).zfill(6)) for i in range(4)]
for name in ['stage3_pick500_replay.py','stage3_pick500_env.py','stage3_pick500_env_smoke.py']:
    ast.parse((S/'src'/name).read_text())
out=O/'native_pilot_v1';assert not out.exists()
command=[spec['python'],'-u',str(S/'src/stage3_pick500_replay.py'),'--headless','--device','cuda:1','--output-dir',str(out)]
for c in candidates:command.extend(['--candidate-dir',str(c)])
commands=[command,[spec['python'],'-u',str(S/'src/stage3_pick500_env_smoke.py'),
    '--headless','--device','cuda:1','--manifest',str(out/'reference_bank.json'),
    '--output-dir',str(O/'object_reset_pilot_v1')]]
(launch/'intent.json').open('x').write(json.dumps({'commands':commands,'scope':'bounded4reference native replay and8env object-reset witness; not formal500reference training'},indent=2))
for index,cmd in enumerate(commands):
    with (launch/('child_'+str(index)+'.log')).open('x') as log:
        child=subprocess.Popen(['timeout','--kill-after=20s','240']+cmd,env={**os.environ,**spec['env']},
            cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (launch/('child_'+str(index)+'.json')).open('x').write(json.dumps({'pid':child.pid}))
        code=child.wait()
    (launch/('exit_'+str(index)+'.json')).open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
    if code!=0:sys.exit(code if code>0 else 1)
    if index==0:
        r=json.loads((out/'report.json').read_text())
        assert r['status']=='native_replay_completed' and r['admitted_references']>=2
replay=json.loads((out/'report.json').read_text());reset=json.loads((O/'object_reset_pilot_v1/report.json').read_text())
result={'status':'native_replay_and_object_reset_passed','admitted_pilot_references':replay['admitted_references'],
    'formal_reference_target':500,'formal_training_started':False,'task_success_claimed':False,
    'object_reset_error_m':reset['maximum_object_reset_error_m'],
    'formal500_count_rejects_pilot':reset['formal500_count_rejected_pilot'],
    'manifest_sha256':hashlib.sha256((out/'reference_bank.json').read_bytes()).hexdigest()}
(launch/'report.json').open('x').write(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
