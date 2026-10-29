"""Explicit source deployment, evidence activation, unique launch and light collect."""
import argparse
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/semantic_evaluator_20261029'
p = argparse.ArgumentParser()
p.add_argument('action', choices=['deploy', 'activate', 'launch', 'collect'])
p.add_argument('--stage', choices=['witnesses', 'train'])
p.add_argument('--role', choices=['author', 'document', 'train'])
p.add_argument('--evidence', required=True)
p.add_argument('--document-summary')
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
sp = B / 'configs/tmr100-capacity-resume-spec.json'
raw = sp.read_bytes()
spec = json.loads(raw)
names = ['scripts/tmr100_capacity_resume_protocol.py', 'scripts/tmr100_capacity_resume_runtime.py',
         'scripts/run_tmr100_capacity_resume.py', 'scripts/supervise_tmr100_capacity_resume.py']
sources = {}
for name in names:
    data = (B / name).read_bytes()
    ast.parse(data, feature_version=(3, 8))
    sources[name] = dict(raw=base64.b64encode(data).decode(), sha256=hashlib.sha256(data).hexdigest())
document = None
if a.action == 'activate' and a.stage == 'train':
    assert a.document_summary
    q = pathlib.Path(a.document_summary)
    workspace = B.parents[3] / 'textop_reproduction_20261024'
    assert q.is_absolute() and q.resolve() == q and q.name == 'SUMMARY.json'
    assert 'textop_reproduction_20261024' in q.parts and not q.is_symlink()
    from filter_recovery_windows_paths import guard_local
    # The owned fresh-agent output is below the task workspace, not a model path.
    task_root = next(parent for parent in q.parents if parent.name == 'textop_reproduction_20261024')
    guard_local(task_root, q)
    document = dict(raw=base64.b64encode(q.read_bytes()).decode(), sha256=hashlib.sha256(q.read_bytes()).hexdigest())
role = a.role or 'author'
run = 'tmr100_capacity_resume_' + role + '_v1'
out = O / (run if a.action in ['launch', 'collect'] else 'tmr100_capacity_resume_preparation_v1')
if a.action == 'launch':
    out.mkdir(exist_ok=False)
else:
    out.mkdir(exist_ok=True)
code = r'''
import ast,base64,fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
action=ACTION_LITERAL;stage=STAGE_LITERAL;role=ROLE_LITERAL;sources=SOURCES_LITERAL;expected=SPEC_SHA_LITERAL
sp=r/'configs/tmr100-capacity-resume-spec.json';newraw=base64.b64decode(SPEC_RAW_LITERAL)
if action=='deploy':
 assert not sp.exists()
 for name,item in sources.items():
  path=r/name;raw=base64.b64decode(item['raw']);assert hashlib_sha(raw)==item['sha256']
  if path.exists():assert hashlib_sha(path.read_bytes())==item['sha256']
  else:
   with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 for name,item in sources.items():ast.parse((r/name).read_bytes(),feature_version=(3,8))
 with sp.open('xb') as f:f.write(newraw);f.flush();os.fsync(f.fileno())
from tmr100_capacity_resume_protocol import ROOT,PARENT,HOME,RUNS,owned,read,sha,origin,workers,closed_job,accepted_witnesses
from contact_filter_resources import sample_resources
assert sha(owned(sp))==expected
s=read(sp);run=RUNS[role];parent=owned(PARENT);prefix='tmr100_capacity_'+run
for name,item in sources.items():assert sha(owned(r/name))==item['sha256']
lock=(r/'receipts/full_library_capacity.lock').open('a+')
if action in ['deploy','activate','launch']:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not workers()
 base,checkpoint=origin(s)
 resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2+16*1024**2)
 gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2;assert gpu>=20*1024**3
 home=pathlib.Path(s['checkpoint_dir']);owned(home.parent,HOME)
 assert subprocess.check_output(['findmnt','-T',str(home.parent),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
 home_required=s['reserve_bytes']+s['log_allowance_bytes']+2*s['checkpoint_upper_bytes']
 assert shutil.disk_usage(home.parent).free>=home_required
 if action=='deploy':
  assert s['witnesses_authorized'] is False and s['formal_training_authorized'] is False
  result=dict(status='TMR100_CAPACITY_RESUME_SOURCES_DEPLOYED_DISABLED',spec_sha256=sha(sp),runtime_binding_sha256=s['runtime_binding_sha256'],resources=resources,gpu1_free_bytes=gpu,home_free_bytes=shutil.disk_usage(home.parent).free,home_required_bytes=home_required,model_constructions=0,formal_updates=0)
 elif action=='activate':
  assert stage in ['witnesses','train']
  if stage=='witnesses':
   assert not s['witnesses_authorized'] and not s['formal_training_authorized']
   assert not home.exists();home.mkdir(mode=0o700);owned(home,HOME)
   probe=home.parent/'capacity_resume_fsync_probe_v1';assert not probe.exists()
   with probe.open('xb') as f:f.write(b'TMR100 resume fsync probe\n');f.flush();os.fsync(f.fileno())
   s['witnesses_authorized']=True
  else:
   assert s['witnesses_authorized'] and not s['formal_training_authorized']
   doc=DOCUMENT_LITERAL;raw=base64.b64decode(doc['raw']);assert hashlib_sha(raw)==doc['sha256']
   summary=json.loads(raw);binding=accepted_witnesses(summary)
   assert binding['runtime_binding_sha256']==s['runtime_binding_sha256']
   dp=r/'receipts/tmr100_capacity_resume_document_summary.json';assert not dp.exists()
   with dp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
   s.update(witnesses_authorized=False,formal_training_authorized=True,witness_bindings=binding,document_summary=dict(path=str(dp),sha256=sha(dp)))
  archive=r/'receipts'/('tmr100_capacity_resume_before_'+stage+'_enable.json');assert not archive.exists()
  with archive.open('xb') as f:f.write(sp.read_bytes());f.flush();os.fsync(f.fileno())
  raw=(json.dumps(s,indent=2)+'\n').encode();partial=sp.with_name(sp.name+'.enable.partial');assert not partial.exists()
  with partial.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  partial.replace(sp)
  request=r/'receipts'/('tmr100_capacity_resume_'+stage+'_spec.json');assert not request.exists()
  with request.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  result=dict(status='TMR100_CAPACITY_RESUME_ENABLED_NOT_DISPATCHED',stage=stage,spec_sha256=sha(sp),spec_base64=base64.b64encode(raw).decode(),runtime_binding_sha256=s['runtime_binding_sha256'],resources=resources)
 elif action=='launch':
  request=owned(r/'receipts'/('tmr100_capacity_resume_'+('train' if role=='train' else 'witnesses')+'_spec.json'));request_spec=read(request)
  assert sha(request)==sha(sp)
  assert request_spec['formal_training_authorized'] if role=='train' else request_spec['witnesses_authorized']
  if role=='document':
   author=closed_job('author');assert author['receipt']['status']=='TMR100_CAPACITY_RESUME_REAL_GPU_WITNESS_VERIFIED'
  if role=='train':assert accepted_witnesses(read(owned(s['document_summary']['path'])))==s['witness_bindings']
  assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
  paths=[r/'receipts'/(prefix+suffix) for suffix in ['_driver.py','_driver.json','_outer_exit.json']];assert not any(p.exists() for p in paths),'Existing intent must be reconciled, never redispatched'
  cmd=['python3',str(r/'scripts/supervise_tmr100_capacity_resume.py'),'--request',str(request),'--role',role]
  driver,info,outer=paths
  text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
  with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
  lock.close()
  with (parent/(run+'_outer.log')).open('xb') as log:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
  start=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
  result=dict(status='TMR100_CAPACITY_RESUME_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,driver_start_ticks=start,request_sha256=sha(request),runtime_binding_sha256=s['runtime_binding_sha256'],resources=resources,gpu1_free_bytes=gpu)
  with info.open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
else:
 assert action=='collect'
 log=parent/(run+'_supervision');files={};hashes={}
 for key,path in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
  if path.exists():files[key]=read(owned(path));hashes[key]=sha(path)
 active=workers()
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
 except BlockingIOError:free=False
 closed=all(key in files for key in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not active and free
 audit=None
 if closed:
  binding=closed_job(role);rec=binding['receipt'];assert rec['runtime_binding_sha256']==s['runtime_binding_sha256']
  if role=='train':
   assert rec['status']=='TMR100_CAPACITY_RESUMED_100_EPOCHS_FINISHED' and rec['formal_optimizer_updates']==101500 and rec['completed_epochs']==100
   assert rec['actual_new_formal_optimizer_updates']==rec['attempted_new_optimizer_updates']==17797 and rec['skipped_updates']==0
   item=rec['selected_checkpoint'];ck=owned(item['path'],HOME);assert ck.parent==pathlib.Path(s['checkpoint_dir']) and ck.name=='tmr_epoch_100_resumed.pth'
   assert ck.stat().st_size==item['bytes']<=201326592 and sha(ck)==item['sha256']
   assert item['production_model_adamw_rng_sampler_nextbatch_reloaded'] and rec['exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload'] and not item['pending_validation']
   audit=dict(actual_resumed_100epochs_101500_exact_fullstate_verified=True,original_safe_source_83703=True,actual_new_formal_updates=17797,original500_success_claimed=False,closure=binding['sha256'])
  else:
   assert rec['status']=='TMR100_CAPACITY_RESUME_REAL_GPU_WITNESS_VERIFIED' and rec['actual_diagnostic_optimizer_calls']==4 and rec['actual_new_formal_optimizer_updates']==0
   compared=False
   if role=='document':accepted_witnesses();compared=True
   audit=dict(actual_resume_fullstate_gpu_loss_replay_verified=True,cross_process_exact=compared,actual_formal_updates=0,closure=binding['sha256'])
 logs={}
 for path in [log/'child.log',parent/(run+'_outer.log')]:
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2400));logs[path.name]=f.read().decode(errors='replace')
 last=None
 if (log/'resources.jsonl').exists():
  with (log/'resources.jsonl').open('rb') as f:f.seek(max(0,(log/'resources.jsonl').stat().st_size-8000));last=json.loads(f.read().decode().splitlines()[-1])
 result=dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=active,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,last_resource=last,logs=logs,data_free_bytes=shutil.disk_usage(r).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)
lock.close();print(json.dumps(result))
'''
# This small helper is inserted before any source writes in the rendered script.
code = 'import hashlib\ndef hashlib_sha(raw):return hashlib.sha256(raw).hexdigest()\n' + code
for key, value in [('ACTION_LITERAL', a.action), ('STAGE_LITERAL', a.stage), ('ROLE_LITERAL', role),
                   ('SOURCES_LITERAL', sources), ('SPEC_SHA_LITERAL', hashlib.sha256(raw).hexdigest()),
                   ('SPEC_RAW_LITERAL', base64.b64encode(raw).decode()), ('DOCUMENT_LITERAL', document)]:
    code = code.replace(key, repr(value))
ast.parse(code, feature_version=(3, 8))
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
if a.action == 'activate':
    updated = base64.b64decode(j['spec_base64'])
    assert hashlib.sha256(updated).hexdigest() == j['spec_sha256']
    with (out / (a.stage + '_spec_before_enable.json')).open('xb') as f:
        f.write(raw)
    sp.write_bytes(updated)
elif a.action == 'collect':
    (out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
elif a.action == 'launch':
    (out / 'dispatch.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
