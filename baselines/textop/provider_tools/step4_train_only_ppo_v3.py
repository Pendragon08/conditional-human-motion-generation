"""Exclusive new PPO author/document/train bridge for original TRAIN only."""
import argparse
import ast
import base64
import hashlib
import json
import os
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O = B/'results/step4_train_only_20261029'
SOURCES = ['scripts/step4_train_only_ppo_runtime_v3.py', 'scripts/step4_train_only_ppo_inputs.py',
    'scripts/step4_train_only_isaaclab_config.py', 'scripts/run_step4_train_only_ppo_v3.py', 'scripts/supervise_step4_train_only_ppo_v3.py']
BOUND = SOURCES+['scripts/step4_train_only_protocol.py', 'scripts/filter_training_runtime.py',
    'scripts/bounded_headless_close.py', 'scripts/run_tracker.sh', 'scripts/solid_floor_diagnostic.py',
    'scripts/contact_filter_resources.py', 'scripts/contact_process_supervision.py', 'configs/tracker-env-spec.json']
TRACKER = 'upstream/TextOpTracker/source/textop_tracker/textop_tracker/'
BOUND += [TRACKER+n for n in ['robots/g1.py', 'tasks/tracking/tracking_env_cfg.py',
    'tasks/tracking/mdp/commands_multi.py', 'tasks/tracking/mdp/observations.py',
    'tasks/tracking/mdp/rewards.py', 'tasks/tracking/mdp/terminations.py',
    'tasks/tracking/config/g1/flat_env_cfg.py', 'tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py']]


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.flush(); os.fsync(f.fileno())


def run_name(role): return 'user_step4_train_only_ppo_'+role+'_v3'


def launch(role, evidence):
    out = O/evidence; out.mkdir(parents=True, exist_ok=False)
    lib = json.loads((O/'library_progress.json').read_bytes())
    assert lib['closed_execution_verified'] and lib['artifact_audit']['all_npz_bytes_sha_CRC_array_hashes_shapes_dtypes_verified']
    assert lib['artifact_audit']['clips'] == 5551
    binding = dict(receipt_sha256=lib['sha256']['receipt'],
        closure_sha256={k: lib['sha256'][k] for k in ['exit', 'process', 'outer']},
        manifest_sha256=lib['files']['receipt']['private_manifest']['sha256'])
    policy = json.loads((B/'configs/step4-original-train-only-ppo-spec.json').read_bytes())
    assert not policy['training_authorized'] and not policy['witnesses_authorized']
    if role in ['document', 'train']:
        author = json.loads((O/'ppo_author_progress.json').read_bytes())
        assert author['run']==run_name('author')
        assert author['closed_execution_verified'] and author['files']['receipt']['recorded_rollout_replay']['real_recorded_rollout_exact_replay']
    if role == 'train':
        document = json.loads((O/'ppo_document_progress.json').read_bytes())
        summary = json.loads((O/'ppo_fresh_document_summary_v3.json').read_bytes())
        assert document['run']==run_name('document')
        assert document['closed_execution_verified']
        assert summary['status'] == 'STAGE4_TRAIN_ONLY_PPO_FRESH_DOCUMENT_VERIFIED' and summary['actual_exit_code'] == 0 and summary['findings'] == []
        assert author['files']['receipt']['initialized_model_sha256'] == document['files']['receipt']['initialized_model_sha256']
        policy.update(witnesses_authorized=False, training_authorized=True,
            accepted_original_train_library=binding, real_RL_witnesses_accepted=True)
    else: policy.update(witnesses_authorized=True, training_authorized=False)
    author_checkpoint = author['files']['receipt']['checkpoints'][0] if role == 'document' else None
    bundle = {n: dict(base64=base64.b64encode((B/n).read_bytes()).decode(), sha256=sha(B/n)) for n in SOURCES}
    for n in SOURCES: ast.parse((B/n).read_bytes(), feature_version=(3, 8))
    binds = {n: sha(B/n) for n in BOUND}
    code = r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
bundle=BUNDLE_LITERAL;binds=BINDS_LITERAL
for n,item in bundle.items():
 raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];ast.parse(raw,feature_version=(3,8))
 p=r/n
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 else:
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
from step4_train_only_protocol import owned,read,sha,atomic,RESERVE,LOG,validate_policy
from step4_train_only_ppo_inputs import accepted_library
from contact_filter_resources import sample_resources
for n,h in binds.items():assert sha(owned(r/n))==h
role=ROLE_LITERAL;run=RUN_LITERAL;policy=POLICY_LITERAL;binding=BINDING_LITERAL;author_checkpoint=AUTHOR_CHECKPOINT_LITERAL
validate_policy(policy);library,rows=accepted_library(binding)
parent=owned(r/'step34_numeric_v1');prefix='step4_train_only_'+run
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_request.json','_outer_exit.json'])
slots=3 if role!='train' else 5;slot=128*1024**2;upper=slots*slot+128*1024**2
resources=sample_resources(r,None,16*1024**3,RESERVE+LOG)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3 and shutil.disk_usage(parent).free>=RESERVE+LOG+upper
s=dict(schema='textop.stage4_train_only_ppo_job.v1',run=run,role=role,policy=policy,library_binding=binding,
 executor_sha256=binds['scripts/run_step4_train_only_ppo_v3.py'],supervisor_sha256=binds['scripts/supervise_step4_train_only_ppo_v3.py'],
 bound_sources=binds,output_root=str(parent),state_slot_bytes=slot,output_upper_bytes=upper,
 author_checkpoint=author_checkpoint,
 source_learning_budget_note='150000 paper-derived candidate; actual target used only for formally accepted new PPO')
request=r/'receipts'/(prefix+'_request.json');atomic(request,s)
cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/supervise_step4_train_only_ppo_v3.py'),'--request',str(request)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
lock.close()
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['/usr/bin/python3','-B',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
st=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text().rsplit(')',1)[1].split()
result=dict(status='STAGE4_TRAIN_ONLY_PPO_DISPATCHED_NOT_ACCEPTED',run=run,role=role,driver_pid=q.pid,driver_start_ticks=st[19],
 request_sha256=sha(request),request=s,resources=resources,gpu_free_bytes=gpu,output_free_bytes=shutil.disk_usage(parent).free)
atomic(r/'receipts'/(prefix+'_driver.json'),result);print(json.dumps(result))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('BUNDLE_LITERAL', repr(bundle)).replace('BINDS_LITERAL', repr(binds)).replace('ROLE_LITERAL', repr(role)).replace('RUN_LITERAL', repr(run_name(role))).replace('POLICY_LITERAL', repr(policy)).replace('BINDING_LITERAL', repr(binding)).replace('AUTHOR_CHECKPOINT_LITERAL', repr(author_checkpoint))
    ast.parse(code, feature_version=(3, 8)); (out/'dispatch_source.py').write_text(code, encoding='utf-8')
    result = json.loads(recorded_remote(code, out/'client.json', timeout=180)); write(out/'dispatch.json', result)
    return result


def collect(role, evidence):
    out = O/evidence; out.mkdir(parents=True, exist_ok=False)
    code = r'''
import fcntl,json,os,pathlib,shutil,sys,time
r=pathlib.Path(ROOT_LITERAL);sys.path.insert(0,str(r/'scripts'))
from step4_train_only_protocol import owned,read,sha
run=RUN_LITERAL;role=ROLE_LITERAL;parent=owned(r/'step34_numeric_v1');prefix='step4_train_only_'+run
sp=owned(r/'receipts'/(prefix+'_request.json'));s=read(sp);log=parent/(run+'_supervision');files={};hashes={}
for k,p in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[k]=read(p);hashes[k]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if str(sp) in argv and any(str(r/'scripts'/n) in argv for n in ['run_step4_train_only_ppo_v3.py','supervise_step4_train_only_ppo_v3.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['process']['child_exit_code']==0 and files['exit']['supervision_passed'] and not workers and free
audit=None
if closed:
 rec=files['receipt'];ex=files['exit'];assert rec['output_writers_closed'] and rec['inputs_unchanged']
 assert rec['status'] in ['STAGE4_TRAIN_ONLY_REAL_PPO_RECOVERY_WITNESS_FINISHED','STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED']
 assert hashes['receipt']==ex['receipt_sha256'] and hashes['process']==ex['process_exit_sha256'] and sha(sp)==ex['request_sha256']
 assert sha(log/'resources.jsonl')==ex['resource_sha256']
 for key in ['exit','process']:
  x=files[key];assert x['child_reaped'] and x['child_exit_code']==0 and x['sample_errors']==[] and not x['timed_out']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[] and samples[-1]['owned_group_gpu_memory']==[]
 for item in rec['checkpoints']:
  path=owned(item['path']);assert path.parent==parent/run and path.stat().st_size==item['bytes']<=s['state_slot_bytes'] and sha(path)==item['sha256']
 if role!='train':
  assert rec['recorded_rollout_replay']['real_recorded_rollout_exact_replay'] and rec['exact_model_Adam_normalizers_all_RNG_ADS_restore_verified']
  assert rec['diagnostic_optimizer_calls']==(60 if role=='document' else 80) and rec['formal_optimizer_updates']==0 and rec['diagnostic_logical_iterations']==2
  if role=='document':assert rec['cross_process_author_checkpoint_restored'] and rec['source_optimizer_updates']==20
 audit=dict(all_checkpoint_bytes_sha_verified=True,private_states=len(rec['checkpoints']),checkpoint_bytes=[x['bytes'] for x in rec['checkpoints']],
  actual_actor431_critic557=True,actual_environment8192=True,actual_full_library5551=True,formal_optimizer_updates=rec['formal_optimizer_updates'])
tails={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));tails[p.name]=f.read().decode(errors='replace')
print(json.dumps(dict(run=run,role=role,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=tails,data_free_bytes=shutil.disk_usage(r).free)))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('RUN_LITERAL', repr(run_name(role))).replace('ROLE_LITERAL', repr(role))
    ast.parse(code, feature_version=(3, 8))
    result = json.loads(recorded_remote(code, out/'client.json', timeout=180)); write(out/'progress.json', result)
    (O/('ppo_'+role+'_progress.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'collect']); p.add_argument('--role', choices=['author','document','train'], required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); assert a.evidence.isascii() and a.evidence.isidentifier()
    print(json.dumps(launch(a.role, a.evidence) if a.action=='launch' else collect(a.role, a.evidence), ensure_ascii=False))
