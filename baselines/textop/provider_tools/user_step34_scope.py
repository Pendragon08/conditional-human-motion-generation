"""Exclusive metadata scope and warm read-only probe for the new user Step3/4.

Never imports a model, simulator or motion array. A scope is a generation plan,
not generated data or authorization to run PPO.
"""
import argparse
import ast
import base64
import hashlib
import json
import os
import pathlib
import shutil
import sys
from backup_filter_recovery_checkpoint import recorded_remote

B=pathlib.Path(__file__).resolve().parents[1]
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O=B/'results/user_step34_20261029'
RUN='user_augmentation5368_scope_v1'
SOURCE_NAMES=['scripts/augmentation5368_protocol.py','scripts/user_step34_contract.py',
    'scripts/build_augmentation5368_scope.py','scripts/supervise_augmentation5368_scope.py',
    'scripts/test_user_step34_contract.py','scripts/user_step4_isaaclab_config.py']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())

def prepare():
    sys.path.insert(0,str(B/'scripts'))
    from user_step34_contract import BASE_TERMS,PRIVILEGED_TERMS,validate
    sources={n:sha(B/n) for n in SOURCE_NAMES}
    for n in SOURCE_NAMES:ast.parse((B/n).read_bytes(),feature_version=(3,8))
    p=B/'configs/user-step4-one-stage-ppo-spec.json'
    policy=dict(schema='textop.user_step4_one_stage_ppo.v1',method='one_stage_PPO',dagger_or_distillation=False,
        actor_terms=[list(x) for x in BASE_TERMS],critic_terms=[list(x) for x in BASE_TERMS+PRIVILEGED_TERMS],
        actor_observation_dim=431,critic_observation_dim=557,future_reference_frames=5,dof=29,action_dim=29,
        actor_hidden=[2048,1024,512],critic_hidden=[2048,1024,512],activation='elu',simulation_hz=200,policy_hz=50,
        rollout_steps_per_env=24,num_envs=8192,learning_rate=.001,minibatches=4,ppo_epochs=5,clip_param=.2,
        seed=20261103,target_RL_iterations=150000,target_iterations_note='Paper real-deployment budget; planned, no RL iterations executed',
        fresh_actor_critic_and_Adam_initialization=True,old_privileged_teacher_training_resumed=False,
        witnesses_authorized=False,training_authorized=False,accepted_mixed_dataset=None,real_RL_witnesses_accepted=False,
        new_RL_training_updates=0,pfail_conditional_regularization=False,domain_randomization=True,
        source_commit='ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22',
        simulator='Warm existing Isaac Lab2.1 / Isaac Sim4.5; no installation or shared-source edits',
        bound_sources={n:sources[n] for n in ['scripts/user_step34_contract.py','scripts/user_step4_isaaclab_config.py']},
        env_spec_sha256=sha(B/'configs/tracker-env-spec.json'),
        blocked_until=['actual5368 generated clips and5666400 control intervals accepted',
            'real and generated references finite and joined by named joints',
            'durable server and private D capacity accepted',
            'TMR/semantic model tasks closed and common model lock free',
            'real one-stage PPO author and independent document recovery witnesses accepted'])
    validate(policy);write(p,policy)
    helper_names=['scripts/filtered15000_vae_completion.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py']
    spec=dict(schema='textop.user_augmentation5368_scope.v1',run=RUN,scope_authorized=True,
        generation_authorized=False,ppo_train_authorized=False,
        executor_sha256=sources['scripts/build_augmentation5368_scope.py'],
        supervisor_sha256=sources['scripts/supervise_augmentation5368_scope.py'],
        bound_sources={n:sha(B/n) for n in SOURCE_NAMES+helper_names},
        train_scope_receipt=dict(path=ROOT+'/evaluator_tmr_20261029/tmr69d_training_scope_v2/receipt.json',
            sha256='f355de1c947487d750cbe3d59c51d78c8853a36315c659a3b08b4876df7273d2'),
        metadata_upper_bytes=64*1024**2,reserve_bytes=2*1024**3,log_allowance_bytes=128*1024**2,
        planned_generated_clips=5368,planned_duration_hours='31.48',planned_control_intervals=5666400,
        actual_generated_clips=0,actual_RL_updates=0)
    write(B/'configs/user-augmentation5368-scope-spec.json',spec)
    return dict(status='STEP34_LOCAL_DISABLED_MODEL_POLICIES_PREPARED',sources=sources,policy_sha256=sha(p),
        request_sha256=sha(B/'configs/user-augmentation5368-scope-spec.json'),new_models=0,new_RL_updates=0)

def launch(evidence):
    out=O/evidence;out.mkdir(parents=True,exist_ok=False)
    names=SOURCE_NAMES+['configs/user-step4-one-stage-ppo-spec.json','configs/user-augmentation5368-scope-spec.json']
    bundle={n:dict(sha256=sha(B/n),base64=base64.b64encode((B/n).read_bytes()).decode()) for n in names}
    code=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from filtered15000_vae_completion import owned,read,sha
run=RUN_LITERAL;bundle=BUNDLE_LITERAL;prefix='step34_'+run
lock=(r/'receipts/user_step34_metadata.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
parent=r/'step34_20261029';parent.mkdir(mode=0o700,exist_ok=True);owned(parent)
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_outer_exit.json'])
for name,item in bundle.items():
 p=r/name;raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 if name.endswith('.py'):ast.parse(raw,feature_version=(3,8))
 if p.exists():assert sha(owned(p))==item['sha256']
 else:
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(owned(p))==item['sha256']
sp=r/'configs/user-augmentation5368-scope-spec.json';s=read(sp)
assert shutil.disk_usage(r).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['metadata_upper_bytes']
for n,h in s['bound_sources'].items():assert sha(owned(r/n))==h
rp=owned(s['train_scope_receipt']['path']);assert sha(rp)==s['train_scope_receipt']['sha256']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
# This isolated CPU test has no data/model/SDK imports. It is never a kernel witness.
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
testlog=parent/'contract_v1';testlog.mkdir(mode=0o700,exist_ok=False)
def sample(pid):return sample_resources(r,pid,16*1024**3,2*1024**3+128*1024**2)
proof=run_supervised(['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
 '/usr/bin/python3','-B',str(r/'scripts/test_user_step34_contract.py')],testlog/'child.log',
 testlog/'resources.jsonl',testlog/'process_exit.json',sample,120,15)
assert proof['supervision_passed']
testresult=json.loads((testlog/'child.log').read_text());assert testresult['negative_cases']==20
lock.close();outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/supervise_augmentation5368_scope.py'),'--request',str(sp)]
text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/(run+'_outer.log')).open('xb') as f:
 q=subprocess.Popen(['/usr/bin/python3','-B',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,env=env,start_new_session=True)
result=dict(status='USER_STEP34_METADATA_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,
 request_sha256=sha(sp),contract_process=proof,contract_result=testresult,model_lock_acquired=False,
 actual_new_models=0,actual_generated_clips=0,actual_RL_updates=0)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('RUN_LITERAL',repr(RUN)).replace('BUNDLE_LITERAL',repr(bundle))
    ast.parse(code,feature_version=(3,8));write(out/'rendered_source_sha256.json',dict(sha256=hashlib.sha256(code.encode()).hexdigest()))
    result=json.loads(recorded_remote(code,out/'client.json',timeout=180));write(out/'dispatch.json',result)
    return result

def collect(evidence):
    out=O/evidence;out.mkdir(parents=True,exist_ok=False)
    code=r'''
import fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(ROOT_LITERAL);sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from filtered15000_vae_completion import owned,read,sha
run=RUN_LITERAL;prefix='step34_'+run;parent=owned(r/'step34_20261029');folder=parent/run;log=parent/(run+'_supervision')
files={}
for k,p in [('receipt',folder/'receipt.json'),('exit',log/'exit.json'),('process',log/'process_exit.json'),('outer',r/'receipts'/(prefix+'_outer_exit.json'))]:
 if p.exists():files[k]=read(owned(p))
ex=files.get('exit',{});rp=folder/'receipt.json'
closed=all(k in files for k in ['receipt','exit','process','outer'])
if closed:
 assert files['outer']['actual_returncode']==0
 for key in ['exit','process']:
  x=files[key];assert x['supervision_passed'] and x['child_exit_code']==0 and x['child_reaped'] and not x['timed_out'] and x['sample_errors']==[]
 assert sha(rp)==ex['receipt_sha256'] and sha(log/'process_exit.json')==ex['process_exit_sha256']
 rows=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert sha(log/'resources.jsonl')==ex['resources_sha256'] and len(rows)==ex['samples_written'] and rows[-1]['phase']=='after_exit'
 assert rows[-1]['owned_group_processes']==[] and rows[-1]['owned_group_gpu_memory']==[]
 item=files['receipt']['private_artifact'];p=owned(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 from augmentation5368_protocol import validate_plan
 plan=read(p);sources={x['initial_source'] for x in plan['clips']}|{x['source_family'] for x in plan['original_RL_clips']}
 texts={x['text_id'] for c in plan['clips'] for x in c['commands']};validate_plan(plan['clips'],sources,texts)
metadata_lock=(r/'receipts/user_step34_metadata.lock').open('a+')
try:fcntl.flock(metadata_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);metadata_free=True
except BlockingIOError:metadata_free=False
metadata_lock.close()
model=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(model,fcntl.LOCK_EX|fcntl.LOCK_NB);model_free=True
except BlockingIOError:model_free=False
model.close()
warm=read(r/'configs/tracker-env-spec.json')
paths=[pathlib.Path(warm[k]) for k in ['isaac_sim_root','isaac_lab_root','robot_urdf']]
assert all(p.exists() for p in paths) and sha(paths[2])==warm['robot_urdf_sha256']
fss={}
for key,path in [('data',r),('home',pathlib.Path('/home/ouyanghao')),('evaluation',pathlib.Path('/home/ouyanghao/textop-user100k-evaluation-20261029'))]:
 assert path.resolve()==path and path.stat().st_uid==1013
 disk=shutil.disk_usage(path);fss[key]=dict(free_bytes=disk.free,total_bytes=disk.total,device_id=path.stat().st_dev,
  mount_info=subprocess.check_output(['findmnt','-n','-o','FSTYPE,TARGET','--target',str(path)],text=True).strip())
training={}
for pid in [416285,416293]:
 p=pathlib.Path('/proc')/str(pid)
 if p.exists() and p.stat().st_uid==1013:
  x=(p/'stat').read_text().rpartition(')')[2].split();training[str(pid)]=dict(starttime=x[19],argv=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode())
for name,h in {'configs/tmr69d-evaluator-train-spec.json':'2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13',
 'scripts/train_tmr69d_evaluator.py':'04014d8f2e657e9c48827edd7fae817c3f01362d97dcdbd3a0e3dea092c604c2'}.items():assert sha(r/name)==h
result=dict(run=run,observed_unix=time.time(),files=files,closed_execution_verified=closed,
 metadata_lock_released=metadata_free,common_model_lock_released=model_free,active_training_identity=training,
 filesystem_capacity=fss,warm_read_only_paths_present=True,robot_urdf_sha256=sha(paths[2]),
 new_SDK_or_model_kernel_executed=False,actual_new_models=0,actual_new_RL_updates=0,
 gpu1_snapshot=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(result))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('RUN_LITERAL',repr(RUN))
    ast.parse(code,feature_version=(3,8));result=json.loads(recorded_remote(code,out/'client.json',timeout=150))
    write(out/'progress.json',result)
    if result['closed_execution_verified']:
        latest=O/RUN;latest.mkdir(exist_ok=True)
        with (latest/'progress.json').open('w',encoding='utf-8') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','collect']);p.add_argument('--evidence');a=p.parse_args()
    O.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':result=prepare()
    else:
        assert a.evidence and a.evidence.isascii() and a.evidence.isidentifier()
        result=globals()[a.action](a.evidence)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
