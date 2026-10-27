"""Read current capacity and active source bindings after the exact cleanup."""
import ast,json,pathlib,shutil,sys
T=pathlib.Path(__file__).resolve().parent;B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
E=B/'results/storage_user_cleanup_20261028'
assert json.loads((E/'server_cleanup.json').read_text())['status']=='OLD_SOFTWARE_INSTALLER_MOVED_TO_PRIVATE_D_VERIFIED'
code=r'''
import hashlib,json,os,pathlib,shutil,subprocess,time
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((r/'configs/filtered15000-vae-spec.json').read_bytes())
assert sha(r/'configs/filtered15000-vae-spec.json')=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
assert sha(r/'scripts/run_filtered15000_vae.py')==s['entrypoint_sha256']
assert sha(r/'scripts/supervise_filtered15000_vae.py')==s['supervisor_sha256']
for n,h in s['bound_sources'].items():assert sha(r/n)==h
ldm=json.loads((r/'configs/filtered15000-ldm-spec.json').read_bytes());ex=json.loads((r/'configs/filtered15000-vae-export-spec.json').read_bytes())
assert not ldm['witness_authorized'] and not ldm['train_authorized'] and not ex['export_authorized']
policy=ldm['checkpoint_policy'];need=policy['reserve_bytes']+policy['log_allowance_bytes']+4*policy['checkpoint_upper_bound_bytes']+ldm['vae_cache_upper_bound_bytes']
export_need=max(ex['minimum_disk_free_bytes'],ex['reserve_bytes']+ex['log_allowance_bytes'])+ex['export_upper_bound_bytes']
def check(path):
 p=pathlib.Path(path);assert p.resolve()==p and p.stat().st_uid==1013
 fs=subprocess.check_output(['findmnt','-T',str(p),'-n','-o','FSTYPE'],text=True,timeout=15).strip();assert fs=='ext4'
 return dict(path=str(p),free_bytes=shutil.disk_usage(p).free,filesystem=fs,owner_uid=1013,writable=os.access(p,os.W_OK|os.X_OK),symlink=False)
data=check(r);home=check(s['checkpoint_dir'])
rec=json.loads((r/'filtered15000_20261027/vae_runs/filtered15000_vae200k_train_v1/receipt.json').read_bytes())
assert rec['status']=='RUNNING_FILTERED15000_VAE_TRAINING' and rec['skipped_updates']==0
current=rec['completed_primitive_attempts'];remaining=sum(x>current for x in s['checkpoint_steps'])+1
vae_needed=s['checkpoint_policy']['reserve_bytes']+s['checkpoint_policy']['log_allowance_bytes']+remaining*s['checkpoint_policy']['checkpoint_upper_bound_bytes']
assert home['free_bytes']>=vae_needed
assert data['free_bytes']>=max(export_need,need+ex['export_upper_bound_bytes'])
old=pathlib.Path('/proc/3237858/status');paused=next((x for x in old.read_text().splitlines() if x.startswith('State:')),'unknown') if old.exists() else 'not present; no signal sent'
print(json.dumps(dict(observed_unix=time.time(),data=data,home=home,ldm_save_and_cache_required_bytes=need,export_upper_bound_bytes=ex['export_upper_bound_bytes'],
 data_required_for_both_export_and_ldm_bytes=max(export_need,need+ex['export_upper_bound_bytes']),both_export_and_ldm_budget_currently_sufficient=True,
 vae_remaining_slot_budget_bytes=vae_needed,vae_remaining_slot_budget_passed=True,vae_actual_step=current,vae_updates=rec['optimizer_updates'],vae_skips=0,
 active_VAE_bound_sources_unchanged=True,old57D_observed_state=paused,formal_LDM_activation_executed=False,capacity_reservation=False,
 new_gpu_contexts=0,signals_sent=False,files_deleted_by_this_probe=0)))
'''
ast.parse(code);j=json.loads(recorded_remote(code,E/'capacity_after_client.json',timeout=90))
d=pathlib.Path('D:/具身智能/TextOp复现_20261024');j['D_free_bytes']=shutil.disk_usage(d).free
j['D_required_for_remaining_three_VAE_upper_bound_states_bytes']=2*1024**3+3*553648128
assert j['D_free_bytes']>=j['D_required_for_remaining_three_VAE_upper_bound_states_bytes']
j['D_remaining_VAE_backup_budget_currently_sufficient']=True
(E/'capacity_after.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))
