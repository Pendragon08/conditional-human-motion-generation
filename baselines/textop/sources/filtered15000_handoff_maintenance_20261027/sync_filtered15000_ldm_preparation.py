"""Incremental public sources/aggregate staging evidence; no private arrays."""
import pathlib,json,hashlib,shutil,sys,os,msvcrt,subprocess,datetime
T=pathlib.Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';B=R/'baselines/textop';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
M=B/'sources/filtered15000_handoff_maintenance_20261027';shutil.copy2(pathlib.Path(__file__),M/pathlib.Path(__file__).name)
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_client
E=T/'filtered15000_ldm_preparation_sync_v1';E.mkdir(exist_ok=False)
names=['.gitattributes', 'README.md', 'CONTINUATION.md', 'FILTERED15000_HANDOFF_ENVIRONMENT.md', 'provider_tools/activate_filtered15000_corpus.py', 'scripts/filtered15000_ldm_runtime.py', 'scripts/full_ldm_history.py', 'configs/filtered15000-ldm-plan.json', 'configs/teacher15000-full-to-generator-autostart-20261027.json', 'reports/filtered15000_ldm_preparation_20261027.md', 'reports/teacher15000_direct_strict_20261027.md', 'results/filtered15000_handoff_tools_20261027/ldm_preparation_cpu_v1.json', 'results/filtered15000_handoff_tools_20261027/ldm_runtime_deploy_v1.json', 'results/teacher15000_direct_strict_20261027/current.json', 'results/teacher15000_direct_strict_20261027/teacher15000_direct_full_v1/progress.json']
names+=sorted(p.relative_to(B).as_posix() for p in M.rglob('*') if p.is_file());assert len(names)==len(set(names))
assert all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 and '.private.' not in n for n in names)
lock=(T/'publish.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
seq=[0]
def git(*args,stdin=''):
 seq[0]+=1;v=recorded_client(stdin,E/('git_'+str(seq[0])+'.json'),timeout=90,command=['git','-C',str(R),'-c','core.quotepath=false','-c','core.whitespace=cr-at-eol',*args],env=env)
 assert v['actual_client_exit_code']==0 and v['client_reaped'] and not v['timed_out'] and v['error'] is None,(args,v['stderr'])
 return v['stdout'].strip()
branch='codex/textop-reproduction-20261024';assert git('branch','--show-current')==branch and not git('diff','--cached','--name-only')
base=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==base
allow={'baselines/textop/'+n for n in names}
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0');assert set(git('diff','--cached','--name-only').splitlines())<=allow
git('diff','--cached','--check');print(git('commit','-m','Implement actual-pass corpus activation and stage full300k DDPM history runtime').splitlines()[0],flush=True)
head=git('rev-parse','HEAD');pack=subprocess.run(['git','-C',str(R),'pack-objects','--stdout','--revs'],input=(head+'\n^'+base+'\n').encode(),capture_output=True,check=True,timeout=60)
assert len(pack.stdout)<5*1024**2
deployed=json.loads((B/'results/filtered15000_handoff_tools_20261027/ldm_runtime_deploy_v1.json').read_text(encoding='utf-8'))['written']
for n,h in deployed.items():
 blob=subprocess.run(['git','-C',str(R),'show',head+':baselines/textop/'+n],capture_output=True,check=True,timeout=30).stdout
 assert hashlib.sha256(blob).hexdigest()==h,n
git('push','--porcelain','origin',branch);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==head
receipt=dict(branch=branch,commit=head,verified_remote_head=remote,url='https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+branch+'/baselines/textop')
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');files=[]
for n in names:
 src=B/n;dst=D/n if n.startswith(('results/','configs/','reports/')) else D/'code/filtered15000_handoff_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
 files.append(dict(source=n,destination=str(dst),bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
result=dict(receipt,files=files,estimated_git_object_pack_bytes=len(pack.stdout),actual_vpn_billed_bytes_unknown=True,private_assets_published=False,
 active_runtime_modified=False,new_training_runs=0,new_gpu_contexts=0,all_new_specs_disabled=True,sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=D/'GitHub同步记录_完整LDM数值入口与语料激活桥_20261027.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(E/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':head,'remote_verified':True,'D_files_sha_verified':len(files),'estimated_git_object_pack_bytes':len(pack.stdout),'new_actual_training':False},ensure_ascii=False))
