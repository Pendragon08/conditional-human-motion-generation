"""Run real online collection/BPTT, stop, reload and finish the exact96-step budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--name',required=True,choices=['online_author_v1','online_doc_v1'])
p.add_argument('--num-envs',type=int,required=True,choices=[4,8])
a=p.parse_args()
D=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103')
spec=json.loads((D/'runtime_bridge_spec.json').read_text())
out=D/'reports'/a.name;out.mkdir(exist_ok=False)
common=[spec['python'],'-u',str(D/'src/stage3_supermimic_pick_train.py'),'--headless','--device','cuda:1',
    '--num-envs',str(a.num_envs),'--macro-rounds','4','--model-warmup-updates','4',
    '--model-batch','16','--policy-batch','16']
commands=[common+['--output-dir',str(out/'prefix'),'--stop-after-vector-steps','72'],
          common+['--output-dir',str(out/'resumed'),'--resume',str(out/'prefix/resume_partial_72.pt')]]
(out/'intent.json').open('x').write(json.dumps({'commands':commands,'runtime_spec':str(D/'runtime_bridge_spec.json'),
    'purpose':'real online loop and full replay/optimizer/RNG/budget resume; not formal task training'},indent=2))
for i,command in enumerate(commands):
    with (out/('child_'+str(i)+'.log')).open('x') as log:
        child=subprocess.Popen(['timeout','--kill-after=20s','240']+command,
            cwd=spec['cwd'],env={**os.environ,**spec['env']},stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/('child_'+str(i)+'.json')).open('x').write(json.dumps({'pid':child.pid}))
        code=child.wait()
    (out/('exit_'+str(i)+'.json')).open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
    if code!=0:sys.exit(code if code>0 else 1)
first=json.loads((out/'prefix/status.json').read_text())
last=json.loads((out/'resumed/status.json').read_text())
assert first['status']=='bounded_prefix_completed' and first['progress']['actual_vector_steps']==72
assert last['status']=='interaction_budget_completed' and last['progress']['actual_vector_steps']==96
assert last['progress']['warmup_included']==64
assert last['progress']['real_environment_transitions']==96*a.num_envs
assert last['counters']['policy_updates']==12 and last['counters']['world_model_updates']==16
assert last['restored_vector_steps']==72 and last['simulator_reinitialized']
assert first['latest_checkpoint']['roundtrip'] and last['latest_checkpoint']['roundtrip']
result={'status':'online_resume_witness_passed','prefix_steps':72,'final_steps':96,
    'warmup_steps_included':64,'actual_real_transitions':96*a.num_envs,
    'world_model_updates':16,'policy_updates':12,'utd_remainder_tenths':last['counters']['utd_tenths'],
    'full_resume_roundtrip':True,'actor_reference_input':False,'formal_training_started':False,
    'task_success_claimed':False,'status_hashes':{x:hashlib.sha256((out/x/'status.json').read_bytes()).hexdigest()
    for x in ['prefix','resumed']}}
(out/'report.json').open('x').write(json.dumps(result,indent=2))
print('SUPERMIMIC_ONLINE_RESUME_PASSED '+json.dumps(result),flush=True)
