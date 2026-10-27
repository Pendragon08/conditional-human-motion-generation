"""Incremental public aggregate evidence sync; private inventory is excluded."""
import datetime,hashlib,json,msvcrt,os,pathlib,shutil,subprocess,sys
T=pathlib.Path(__file__).resolve().parent
R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation'
B=R/'baselines/textop';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
M=B/'sources/filtered15000_handoff_maintenance_20261027'
for name in ['observe_filtered15000_storage_recovered.py','inventory_filtered15000_owned_software.py','reconcile_filtered15000_access_recovered.py','sync_filtered15000_remote_recovered.py']:
 shutil.copy2(T/name,M/name)
sys.path.insert(0,str(B/'provider_tools'))
from backup_filter_recovery_checkpoint import recorded_client
E=T/'filtered15000_remote_recovered_sync_v1';E.mkdir(exist_ok=False)
P='results/filtered15000_handoff_20261027/filtered15000_vae200k_train_v1/'
names=['CONTINUATION.md','README.md','FILTERED15000_HANDOFF_ENVIRONMENT.md',
 'results/teacher15000_direct_strict_20261027/current.json','reports/teacher15000_direct_strict_20261027.md',
 'reports/filter_first_restored_20261027.md','configs/teacher15000-direct-strict-priority-20261027.json',
 'reports/filtered15000_remote_recovery_20261028.md',P+'access_status.json',
 P+'access_status_before_recovery_20261028_0557.json',P+'progress.json',P+'heartbeat_20261027_2156.json',
 P+'heartbeat_20261027_2156_client.json',P+'heartbeat_20261027_2156_client_intent.json',
 'results/filtered15000_handoff_20261027/vae_eta_recovery_20261028_0557.json',
 'results/filtered15000_handoff_20261027/storage_recovered_20261028_v1/observation.json']
names+=sorted(p.relative_to(B).as_posix() for p in M.rglob('*') if p.is_file())
assert len(names)==len(set(names))
assert all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 and '.private.' not in n and not n.endswith('software_inventory.json') for n in names)
lock=(T/'publish.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
seq=[0]
def git(*args,stdin=''):
 seq[0]+=1
 v=recorded_client(stdin,E/('git_'+str(seq[0])+'.json'),timeout=90,command=['git','-C',str(R),'-c','core.quotepath=false','-c','core.whitespace=cr-at-eol',*args],env=env)
 assert v['actual_client_exit_code']==0 and v['client_reaped'] and not v['timed_out'] and v['error'] is None,(args,v['stderr'])
 return v['stdout'].strip()
branch='codex/textop-reproduction-20261024'
assert git('branch','--show-current')==branch and not git('diff','--cached','--name-only')
base=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==base
allow={'baselines/textop/'+n for n in names}
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow
git('diff','--cached','--check')
print(git('commit','-m','Verify recovered VAE observation and measure pending LDM storage').splitlines()[0],flush=True)
head=git('rev-parse','HEAD')
pack=subprocess.run(['git','-C',str(R),'pack-objects','--stdout','--revs'],input=(head+'\n^'+base+'\n').encode(),capture_output=True,check=True,timeout=60)
assert len(pack.stdout)<5*1024**2
for n in ['configs/filtered15000-vae-spec.json','scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py','scripts/generator8278_vae_runtime.py']:
 blob=subprocess.run(['git','-C',str(R),'show',head+':baselines/textop/'+n],capture_output=True,check=True,timeout=30).stdout
 assert hashlib.sha256(blob).digest()==hashlib.sha256((B/n).read_bytes()).digest(),n
git('push','--porcelain','origin',branch)
remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==head
receipt=dict(branch=branch,commit=head,verified_remote_head=remote,url='https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+branch+'/baselines/textop')
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
files=[]
for n in names:
 src=B/n;dst=D/n if n.startswith(('results/','configs/','reports/')) else D/'code/filtered15000_handoff_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
 files.append(dict(source=n,destination=str(dst),bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
result=dict(receipt,files=files,estimated_git_object_pack_bytes=len(pack.stdout),actual_vpn_billed_bytes_unknown=True,
 current_remote_training_state_verified=True,last_verified_primitive_attempts=41400,optimizer_updates=41400,skips=0,
 new_formal_training_runs=0,new_gpu_contexts=0,training_source_modified=False,files_deleted=0,
 private_directory_inventory_published=False,private_checkpoint_assets_published=False,
 ldm_capacity_accepted=False,private_checkpoint_backups_created=0,sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(D/'GitHub同步记录_远程恢复实核与后续容量_20261028.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(E/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(commit=head,remote_verified=True,D_files_sha_verified=len(files),estimated_git_object_pack_bytes=len(pack.stdout),actual_new_training=False),ensure_ascii=False))
