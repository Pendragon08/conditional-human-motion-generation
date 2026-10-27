"""Publish bounded recovery code/aggregate evidence before/after exact cleanup."""
import argparse,datetime,hashlib,json,msvcrt,os,pathlib,shutil,subprocess,sys
T=pathlib.Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';B=R/'baselines/textop'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024');S=B/'results/storage_user_cleanup_20261028'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_client
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pre','post'],required=True);p.add_argument('--attempt',default='v1');a=p.parse_args()
assert a.attempt.isascii() and a.attempt.isalnum()
E=T/('user_cleanup_publish_'+a.phase+'_'+a.attempt);E.mkdir(exist_ok=False)
M=B/'sources/filtered15000_handoff_maintenance_20261027';shutil.copy2(__file__,M/pathlib.Path(__file__).name)
names=['provider_tools/'+n for n in ['verify_cleanup_candidates_20261028.py','probe_idle_installer_20261028.py','restore_contact_transport_part_20261028.py','execute_user_cleanup_20261028.py','remove_verified_transport_parts_20261028.ps1']]
names+=['sources/filtered15000_handoff_maintenance_20261027/'+pathlib.Path(__file__).name]
names+=['results/storage_user_cleanup_20261028/'+x.name for x in S.glob('*.json')]
names+=['reports/user_authorized_storage_cleanup_20261028.md']
if a.phase=='post':
 names+=['CONTINUATION.md','README.md','FILTERED15000_HANDOFF_ENVIRONMENT.md','results/teacher15000_direct_strict_20261027/current.json','reports/teacher15000_direct_strict_20261027.md','reports/filter_first_restored_20261027.md','configs/teacher15000-direct-strict-priority-20261027.json']
 P='results/filtered15000_handoff_20261027/filtered15000_vae200k_train_v1/'
 names+=[P+'progress.json',P+'user_cleanup_post_20261028.json',P+'user_cleanup_post_20261028_client.json',P+'user_cleanup_post_20261028_client_intent.json']
assert len(names)==len(set(names)) and all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 and '.private.' not in n for n in names)
lock=(T/'publish.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0');seq=[0]
def git(*args,stdin=''):
 seq[0]+=1;v=recorded_client(stdin,E/('git_'+str(seq[0])+'.json'),timeout=90,command=['git','-C',str(R),'-c','core.quotepath=false','-c','core.whitespace=cr-at-eol',*args],env=env)
 assert v['actual_client_exit_code']==0 and v['client_reaped'] and not v['timed_out'] and v['error'] is None,(args,v['stderr'])
 return v['stdout'].strip()
branch='codex/textop-reproduction-20261024';assert git('branch','--show-current')==branch and not git('diff','--cached','--name-only')
base=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==base
allow={'baselines/textop/'+n for n in names};git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow;git('diff','--cached','--check')
print(git('commit','-m',('Preserve recovery proofs before user-authorized exact-copy cleanup' if a.phase=='pre' else 'Verify user-authorized space recovery without changing VAE training')).splitlines()[0],flush=True)
head=git('rev-parse','HEAD');pack=subprocess.run(['git','-C',str(R),'pack-objects','--stdout','--revs'],input=(head+'\n^'+base+'\n').encode(),capture_output=True,check=True,timeout=60)
assert len(pack.stdout)<5*1024**2
for n in ['configs/filtered15000-vae-spec.json','scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py','scripts/generator8278_vae_runtime.py']:
 blob=subprocess.run(['git','-C',str(R),'show',head+':baselines/textop/'+n],capture_output=True,check=True,timeout=30).stdout;assert hashlib.sha256(blob).digest()==hashlib.sha256((B/n).read_bytes()).digest()
git('push','--porcelain','origin',branch);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==head
receipt=dict(branch=branch,commit=head,verified_remote_head=remote,url='https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+branch+'/baselines/textop')
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');files=[]
for n in names:
 src=B/n;dst=D/n if n.startswith(('results/','configs/','reports/')) else D/'code/filtered15000_handoff_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes();files.append(dict(name=n,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
assert shutil.disk_usage(D).free>=2*1024**3
result=dict(receipt,phase=a.phase,files=files,estimated_git_object_pack_bytes=len(pack.stdout),actual_vpn_billed_bytes_unknown=True,private_payloads_published=False,
 active_training_code_modified=False,new_gpu_contexts=0,cleanup_executed_by_publish=False,D_free_bytes=shutil.disk_usage(D).free)
(E/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(D/('GitHub同步记录_用户授权空间清理_'+a.phase+'_20261028.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(commit=head,remote_verified=True,phase=a.phase,D_sha_verified_files=len(files),estimated_git_object_pack_bytes=len(pack.stdout))))
