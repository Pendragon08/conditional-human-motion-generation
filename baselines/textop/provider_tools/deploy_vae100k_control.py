"""Deploy new disabled CPU control sources; never edit the active VAE program."""
import ast,base64,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O=B/'results/vae100k_user_handoff_20261028';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
original=json.loads((B/'configs/filtered15000-vae-spec.json').read_bytes())
assert sha(B/'configs/filtered15000-vae-spec.json')=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
names=['scripts/vae100k_control_protocol.py','scripts/watch_vae100k_handoff.py','scripts/supervise_vae100k_handoff.py','provider_tools/test_vae100k_control_cpu.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
bound=['scripts/vae100k_control_protocol.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py',
 'configs/filtered15000-vae-spec.json','scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py',
 'scripts/generator8278_vae_runtime.py','scripts/generator_training_state.py']
producer=dict(pid=165381,start_ticks='1432055933',uid=1013,argv=[ROOT+'/env/bin/python',ROOT+'/scripts/run_filtered15000_vae.py','--mode','train','--run','filtered15000_vae200k_train_v1'])
supervisor=dict(pid=165374,start_ticks='1432055921',uid=1013,argv=['python3',ROOT+'/scripts/supervise_filtered15000_vae.py','--mode','train','--run','filtered15000_vae200k_train_v1'])
s=dict(schema='textop.user_vae100k_control.v1',control_authorized=False,
 latest_user_request='训练到100000步就行了，然后训练LDM。',target_primitive_attempts=100000,
 original_run='filtered15000_vae200k_train_v1',original_spec_sha256=sha(B/'configs/filtered15000-vae-spec.json'),
 original_entry_sha256=original['entrypoint_sha256'],original_runtime_binding_sha256='87a2f61683d930bf5fe69a9113713c90137c43318766627975c6d11accc20c1c',
 checkpoint_path=original['checkpoint_dir']+'/vae_state_100000.pth',checkpoint_allowed_root=original['checkpoint_allowed_root'],
 checkpoint_upper_bound_bytes=original['checkpoint_policy']['checkpoint_upper_bound_bytes'],producer=producer,supervisor=supervisor,
 entrypoint_sha256=sha(B/names[1]),supervisor_sha256=sha(B/names[2]),bound_sources={n:sha(B/n) for n in bound},
 one_signal_only=True,kill_signal_authorized=False,original_source_edit_authorized=False,
 require_before_signal=['real100000 named full state','producer exact reload receipt','matching frozen sources/data','live pidfd identity'],
 early_stop_is_user_variant_not_200k_completion=True)
for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:s[k]=original[k]
sp=B/'configs/vae100k-user-control-spec.json';raw=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode();sp.write_bytes(raw)
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names};payload['configs/vae100k-user-control-spec.json']=base64.b64encode(raw).decode()
code=r'''
import base64,hashlib,json,os,pathlib,shutil,subprocess,fcntl
r=pathlib.Path(ROOT);assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});items=PAYLOAD;s=POLICY
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['bound_sources'].items():
 if n!='scripts/vae100k_control_protocol.py':assert sha(r/n)==h
assert shutil.disk_usage(r).free>2*1024**3
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>=1024**3
workers=[]
for w in [s['producer'],s['supervisor']]:
 p=pathlib.Path('/proc')/str(w['pid']);fields=(p/'stat').read_text().split(') ',1)[1].split()
 actual=dict(pid=w['pid'],start_ticks=fields[19],uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
 assert actual==w;workers.append(actual)
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:busy=True
else:busy=False
lock.close();assert busy,'Original VAE must still own its GPU lock'
for n,b in items.items():
 p=r/n;assert not p.exists() and p.parent.resolve()==p.parent
 p.parent.mkdir(parents=True,exist_ok=True);assert p.parent.stat().st_uid==1013
 with p.open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
assert all(sha(r/n)==h for n,h in s['bound_sources'].items())
out=dict(status='NEW_VAE100K_CONTROL_DEPLOYED_DISABLED',files_sha256={n:sha(r/n) for n in items},workers=workers,
 original_GPU_lock_busy=True,new_gpu_contexts=0,production_signals_sent=0,original_training_source_modified=False)
print(json.dumps(out))
'''
code=code.replace('ROOT',repr(ROOT)).replace('PAYLOAD',repr(payload)).replace('POLICY',repr(s));ast.parse(code)
result=json.loads(recorded_remote(code,O/'control_deploy_v1_client.json',timeout=90))
(O/'control_deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
