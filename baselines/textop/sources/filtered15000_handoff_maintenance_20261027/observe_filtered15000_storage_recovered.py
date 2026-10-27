"""Read-only storage observations after SSH recovery; never operate training."""
import ast, datetime, hashlib, json, pathlib, shutil, sys

T = pathlib.Path(__file__).resolve().parent
B = T.parent / 'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
sys.path.insert(0, str(B/'provider_tools'))
from backup_filter_recovery_checkpoint import recorded_remote

E = B/'results/filtered15000_handoff_20261027/storage_recovered_20261028_v1'
E.mkdir(exist_ok=False)
code = r'''
import datetime,hashlib,json,os,pathlib,shutil,subprocess,time
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
home=pathlib.Path('/home/ouyanghao')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
specpath=r/'configs/filtered15000-vae-spec.json'
assert sha(specpath)=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
s=json.loads(specpath.read_bytes())
ldm=json.loads((r/'configs/filtered15000-ldm-spec.json').read_bytes())
exp=json.loads((r/'configs/filtered15000-vae-export-spec.json').read_bytes())
assert not ldm['witness_authorized'] and not ldm['train_authorized'] and not exp['export_authorized']
def measure(p):
 assert p.exists() and p.resolve()==p and p.stat().st_uid==1013
 fs=subprocess.check_output(['findmnt','-T',str(p),'-n','-o','FSTYPE'],text=True,timeout=15).strip()
 st=p.stat()
 return dict(path=str(p),owner_uid=st.st_uid,device=st.st_dev,filesystem=fs,free_bytes=shutil.disk_usage(p).free,symlink=False,writable=os.access(p,os.W_OK|os.X_OK))
dest=pathlib.Path(s['checkpoint_dir'])
files=[]
for p in sorted(dest.iterdir()):
 st=p.lstat();files.append(dict(name=p.name,bytes=st.st_size,mtime_ns=st.st_mtime_ns,owner_uid=st.st_uid,symlink=p.is_symlink()))
receipt=r/'filtered15000_20261027/vae_runs/filtered15000_vae200k_train_v1/receipt.json'
rec=json.loads(receipt.read_bytes())
policy=ldm['checkpoint_policy']
needed=policy['reserve_bytes']+policy['log_allowance_bytes']+4*policy['checkpoint_upper_bound_bytes']+ldm['vae_cache_upper_bound_bytes']
export_needed=max(exp['minimum_disk_free_bytes'],exp['reserve_bytes']+exp['log_allowance_bytes'])+exp['export_upper_bound_bytes']
data=measure(r);h=measure(home);checkpoint=measure(dest)
out=dict(schema='textop.filtered15000_readonly_storage_observation.v1',observed_unix=time.time(),
 vae_spec_sha256=sha(specpath),vae_entry_sha256=sha(r/'scripts/run_filtered15000_vae.py'),
 vae_numeric_sha256=sha(r/'scripts/generator8278_vae_runtime.py'),
 vae_progress=dict(status=rec['status'],step=rec['completed_primitive_attempts'],updates=rec['optimizer_updates'],skips=rec['skipped_updates'],receipt_sha256=sha(receipt)),
 data=data,home=h,vae_checkpoint_directory=checkpoint,checkpoint_entries=files,
 ldm_required_bytes_including_cache=needed,export_required_bytes=export_needed,
 data_free_after_export_upper_bound=data['free_bytes']-exp['export_upper_bound_bytes'],
 data_currently_sufficient_for_export_and_ldm=data['free_bytes']>=max(export_needed,needed+exp['export_upper_bound_bytes']),
 storage_acceptance=False,ldm_launch_enabled=False,new_model_loads=0,new_gpu_contexts=0,new_sdk_tasks=0,
 training_source_changed=False,signals_sent=False,network_changed=False,files_deleted=0,
 note='Read-only sample, no reservation or future capacity guarantee; no checkpoint payload read and no common-lock acquisition')
print(json.dumps(out))
'''
ast.parse(code)
result = json.loads(recorded_remote(code,E/'storage_client.json',timeout=90))
result['D'] = dict(path=str(D),free_bytes=shutil.disk_usage(D).free,reserve_bytes=2*1024**3)
bound=553648128
result['D']['can_fit_one_upper_bound_checkpoint_with_2GiB_reserve'] = result['D']['free_bytes']>=2*1024**3+bound
result['D']['can_fit_all_four_upper_bound_checkpoints_with_2GiB_reserve'] = result['D']['free_bytes']>=2*1024**3+4*bound
(E/'observation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
