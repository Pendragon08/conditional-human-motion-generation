"""Deploy a new disabled stdlib observer; leave current LDM math/source frozen."""
import ast,base64,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/ldm100k_user_handoff_20261028';O.mkdir(exist_ok=False)
j=json.loads((B/'results/filtered15000_handoff_20261027/user_vae100k_ldm300k_train_v1/progress.json').read_bytes());r=j['files']['receipt'];assert r['status']=='RUNNING_FILTERED15000_LDM_TRAINING'
sp0=B/'configs/user-vae100k-ldm-spec.json';original=json.loads(sp0.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(sp0)==r['spec_sha256']=='654332646f2aa4bb69fea4cc779edf88138735a0ca9f5e5906e783948868c770'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
def worker(script):
 x=next(x for x in j['workers'] if ROOT+'/scripts/'+script in x['argv'])
 return dict(pid=x['pid'],start_ticks=x['start_ticks'],uid=1013,argv=[a for a in x['argv'] if a])
names=['scripts/ldm100k_control_protocol.py','scripts/watch_ldm100k_handoff.py','scripts/supervise_ldm100k_handoff.py','provider_tools/test_ldm100k_control_cpu.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'),feature_version=(3,8))
bound=list(original['bound_sources'])+['configs/user-vae100k-ldm-spec.json','scripts/run_user_vae100k_ldm.py','scripts/supervise_user_vae100k_ldm.py','scripts/ldm100k_control_protocol.py','scripts/vae100k_control_protocol.py','scripts/quick_pidfd.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py']
s=dict(schema='textop.user_ldm100k_control.v1',control_authorized=False,latest_user_request='LDM也训练100000步吧，不需要300000步。',target_primitive_attempts=100000,original_run=j['run'],original_spec_sha256=sha(sp0),original_entry_sha256=original['entrypoint_sha256'],original_runtime_binding_sha256=r['runtime_binding_sha256'],checkpoint_path=original['checkpoint_dir']+'/ldm_state_100000.pth',checkpoint_allowed_root=original['checkpoint_allowed_root'],checkpoint_upper_bound_bytes=original['checkpoint_policy']['checkpoint_upper_bound_bytes'],producer=worker('run_user_vae100k_ldm.py'),supervisor=worker('supervise_user_vae100k_ldm.py'),entrypoint_sha256=sha(B/names[1]),supervisor_sha256=sha(B/names[2]),bound_sources={n:sha(B/n) for n in sorted(set(bound))},one_signal_only=True,kill_signal_authorized=False,original_source_edit_authorized=False,quality_is_audited_after_stop_not_permission_to_exceed_target=True,original_300k_learning_schedule_preserved=True)
for k in ['vae_checkpoint_sha256','vae_inference_sha256','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:s[k]=r[k]
sp=B/'configs/ldm100k-user-control-spec.json';assert not sp.exists();raw=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode();sp.write_bytes(raw)
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names};payload['configs/ldm100k-user-control-spec.json']=base64.b64encode(raw).decode()
code=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});s=POLICY;items=PAYLOAD;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['bound_sources'].items():
 if n not in items:assert sha(r/n)==h,n
sys.path.insert(0,str(r/'scripts'))
from vae100k_control_protocol import verify_identity
verify_identity(s['producer']);verify_identity(s['supervisor'])
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:busy=True
else:busy=False
lock.close();assert busy
assert shutil.disk_usage(r).free>2*1024**3
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>1024**3
for n,b in items.items():
 p=r/n;assert not p.exists() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 raw=base64.b64decode(b)
 if n.endswith('.py'):ast.parse(raw,filename=n)
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
print(json.dumps(dict(status='NEW_LDM100K_CONTROL_DEPLOYED_DISABLED',files_sha256={n:sha(r/n) for n in items},original_ldm_source_modified=False,production_signals_sent=0,new_gpu_contexts=0,new_model_loads=0)))
'''.replace('POLICY',repr(s)).replace('PAYLOAD',repr(payload))
ast.parse(code);z=json.loads(recorded_remote(code,O/'deploy_client.json',timeout=90));(O/'deploy.json').write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z))
