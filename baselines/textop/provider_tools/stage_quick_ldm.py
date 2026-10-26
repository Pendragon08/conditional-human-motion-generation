"""Stage a disabled-training LDM profile only after real VAE completion/backup."""
import ast,base64,hashlib,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
names=['scripts/quick_ldm_runtime.py','scripts/run_quick_ldm.py','scripts/supervise_quick_ldm.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
backups=json.loads((B/'results/quick_vae_20261026/quick_vae_train_v1/checkpoint_backup.json').read_text(encoding='utf-8'))['checkpoints']
ck=[x for x in backups if x['manager_step']==2000];assert len(ck)==1 and ck[0]['private_server_and_local_sha_verified']
f=D/'checkpoints/private/quick_vae_20261026/vae_state_2000.pth';assert f.stat().st_size==ck[0]['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==ck[0]['sha256']
script=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.getuid()==1013
out=r/'receipts/quick_ldm_stage_v1.json';assert not out.exists()
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rp=r/'quick_generator_20261026/vae_runs/quick_vae_train_v1/receipt.json';ep=r/'receipts/quick_vae_quick_vae_train_v1_exit.json';vr=json.loads(rp.read_text());ve=json.loads(ep.read_text())
assert ve['supervision_passed'] and ve['child_exit_code']==0 and ve['evaluation_receipt_sha256']==sha(rp)
assert vr['status']=='QUICK_VAE_2000_TRAINING_FINISHED' and vr['output_writers_closed'] and vr['vae_quality_gate_passed'] and vr['completed_primitive_attempts']==2000
assert vr['latest_file_resume_verification']['rng_and_generator_restored_exactly']
ck=[x for x in vr['checkpoints'] if x['manager_step']==2000];assert len(ck)==1 and ck[0]['sha256']==BACKUP_SHA and sha(ck[0]['path'])==BACKUP_SHA
plan=json.loads((r/'configs/quick-reproduction-plan-20261026.json').read_text());frozen=dict(plan['frozen_runtime_files'],**plan['control_sources'])
assert all(sha(r/n)==h for n,h in frozen.items())
home=pathlib.Path('/home/ouyanghao/textop-quick69-checkpoints-20261026');assert home==home.resolve() and not home.is_symlink();home.mkdir(exist_ok=True)
dest=home/'ldm';dest.mkdir(exist_ok=True)
assert home.stat().st_uid==dest.stat().st_uid==1013 and os.access(dest,os.W_OK)
assert subprocess.check_output(['findmnt','-T',str(dest),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
assert shutil.disk_usage(dest).free>=2*1024**3+512*1024**2+4*300*1024**2+600*1024**2
payload=PAYLOAD
for n,b in payload.items():
 p=r/n;raw=base64.b64decode(b);assert not p.exists();ast.parse(raw.decode());p.write_bytes(raw)
bound=['scripts/paper69_runtime.py','scripts/paper69_data_runtime.py','scripts/generator_training_state.py','scripts/checkpoint_storage_guard.py','scripts/quick_vae_runtime.py','scripts/quick_ldm_runtime.py','configs/paper69-env-spec.json']
spec={'schema':'textop.quick_ldm_runtime.v1','witness_authorized':True,'train_authorized':False,'witnesses':[],'seed':20261026,
 'data_acceptance_sha256':vr['data_acceptance_sha256'],'corpus_receipt_sha256':vr['corpus_receipt_sha256'],'corpus_dir':str(r/'quick_generator_20261026/corpus_v1'),
 'vae_training':{'receipt_path':str(rp),'receipt_sha256':sha(rp),'exit_path':str(ep),'exit_sha256':sha(ep)},'vae_checkpoint':ck[0],
 'checkpoint_dir':str(dest),'checkpoint_allowed_root':str(home),'vae_cache_upper_bound_bytes':600*1024**2,
 'checkpoint_policy':{'reserve_bytes':2*1024**3,'log_allowance_bytes':512*1024**2,'checkpoint_upper_bound_bytes':300*1024**2},
 'ldm':{'stages':[1000,1000,1000],'batch_size':512,'learning_rate':.0001,'latent_shape':[1,128],'hidden':512,'ff':1024,'layers':8,'heads':4,'feature_dim':69,'history':2,'future':8,'primitives':4,'clip_dim':512,'cond_mask_prob':.1,'diffusion_steps':5,'noise_schedule':'cosine','prediction':'x0','cfg':5,'ema':False},
 'bound_sources':{n:sha(r/n) for n in bound},'entrypoint_sha256':sha(r/'scripts/run_quick_ldm.py'),'supervisor_sha256':sha(r/'scripts/supervise_quick_ldm.py'),
 'scope':'Quick3000 attempts; compressed curriculum/LR clocks, not paper300k. Actual frozen VAE2000 required; shared cache budget explicit; no fullpaper/closedloop claim.'}
sp=r/'configs/quick-ldm-runtime-spec.json';assert not sp.exists();sp.write_text(json.dumps(spec,indent=2)+'\n')
assert all(sha(r/n)==h for n,h in frozen.items())
proof={'status':'QUICK_LDM_PROFILE_STAGED_TRAINING_DISABLED','spec_sha256':sha(sp),'spec_base64':base64.b64encode(sp.read_bytes()).decode(),'source_sha256':{n:sha(r/n) for n in payload},'vae_checkpoint':ck[0],'home_free_bytes':shutil.disk_usage(dest).free,'old19_runtime_unchanged':True,'new_optimizer_updates':0,'new_sdk_runs':0}
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''.replace('BACKUP_SHA',repr(ck[0]['sha256'])).replace('PAYLOAD',repr(payload))
j=json.loads(remote(script,timeout=90));(B/'configs/quick-ldm-runtime-spec.json').write_bytes(base64.b64decode(j['spec_base64']))
out=B/'results/quick_ldm_20261026';out.mkdir(exist_ok=True);(out/'stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in j.items() if k not in ['spec_base64','source_sha256']}))
