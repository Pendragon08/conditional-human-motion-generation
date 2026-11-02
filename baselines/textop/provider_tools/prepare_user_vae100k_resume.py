"""Recover actual99700 capacity stop, preserve originals, enable bounded witnesses."""
import ast,base64,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028'
j=json.loads((B/'results/filtered15000_handoff_20261027/filtered15000_vae200k_train_v1/progress.json').read_bytes());r=j['files']['receipt'];assert not j['workers'] and j['common_lock_free']
assert r['completed_primitive_attempts']==99700 and r['emergency_checkpoint']['manager_step']==99700
old=json.loads((B/'configs/filtered15000-vae-spec.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['scripts/user_vae100k_resume_protocol.py','scripts/run_user_vae100k_resume.py','scripts/supervise_user_vae100k_resume.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'),feature_version=(3,8))
bound={n:sha(B/n) for n in set(list(old['bound_sources'])+names+['scripts/vae100k_control_protocol.py','scripts/quick_pidfd.py','scripts/filtered15000_vae_completion.py','scripts/generator8278_vae_runtime.py','scripts/generator_training_state.py','scripts/contact_filter_resources.py','scripts/contact_process_supervision.py','scripts/checkpoint_storage_guard.py'])}
s=dict(schema='textop.user_vae100k_capacity_resume.v1',latest_user_request='训练到100000步就行了，然后训练LDM。',original_spec_sha256=sha(B/'configs/filtered15000-vae-spec.json'),original_receipt_sha256=j['sha256']['receipt'],original_exit_sha256={k:j['sha256'][k] for k in ['exit','process','outer']},original_runtime_binding_sha256=r['runtime_binding_sha256'],resume_checkpoint=r['emergency_checkpoint'],target_checkpoint_path=old['checkpoint_dir']+'/vae_state_100000.pth',entrypoint_sha256=sha(B/names[1]),supervisor_sha256=sha(B/names[2]),bound_sources=bound,witness_authorized=True,train_authorized=False,witnesses=[],new_formal_optimizer_updates=300,original_max_steps_and_lr_denominator_preserved=200000,actual_user_endpoint=100000,checkpoint_slots_required_for_continuation=2,old_failure_preserved=True)
sp=B/'configs/user-vae100k-resume-spec.json';assert not sp.exists();raw=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode();sp.write_bytes(raw)
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names};payload['configs/user-vae100k-resume-spec.json']=base64.b64encode(raw).decode()
code=r'''
import ast,base64,fcntl,gzip,hashlib,json,os,pathlib,sys,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});s=__SPEC__;items=__PAYLOAD__
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['bound_sources'].items():
 if n not in items:assert sha(r/n)==h,n
assert sha(r/'configs/filtered15000-vae-spec.json')==s['original_spec_sha256']
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,b in items.items():
 p=r/n;assert not p.exists() and not p.is_symlink() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 if n.endswith('.py'):ast.parse(base64.b64decode(b),filename=n)
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>16*1024**3 and shutil.disk_usage(r).free>3*1024**3
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
sys.path.insert(0,str(r/'scripts'))
from user_vae100k_resume_protocol import failed_source
from checkpoint_storage_guard import check
v=failed_source(s);original=v['spec'];capacity=check(pathlib.Path(original['checkpoint_dir']),pathlib.Path(original['checkpoint_allowed_root']),original['checkpoint_policy'],remaining_saves=2)
metrics=r/'filtered15000_20261027/vae_runs/filtered15000_vae200k_train_v1/metrics.jsonl.gz';expanded=0
with gzip.open(metrics,'rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):expanded+=len(b);assert expanded<512*1024**2
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in __import__('subprocess').check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
print(json.dumps(dict(status='REAL99700_SOURCE_AND_CAPACITY_VERIFIED_RESUME_WITNESSES_ENABLED_NOT_DISPATCHED',files_sha256={n:sha(r/n) for n in items},source_fingerprint=v['checkpoint_proof'],original_metrics_crc_closed=True,metrics_uncompressed_bytes=expanded,capacity=capacity,new_gpu_contexts=0,new_model_loads=0,signals_sent=0)))
'''
code=code.replace('__SPEC__',repr(s)).replace('__PAYLOAD__',repr(payload));ast.parse(code)
z=json.loads(recorded_remote(code,O/'rdeploy_client.json',timeout=180));(O/'rdeploy.json').write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z))
