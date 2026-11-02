"""Explicit small incremental public sync; no raw data or release reupload."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,msvcrt
T=Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';B=R/'baselines/textop'
D=Path('D:/具身智能/TextOp复现_20261024');BRANCH='codex/textop-reproduction-20261024'
# Retain current sync machinery as source, not current executable entrypoints.
M=B/'sources/sync_maintenance_20261027';M.mkdir(exist_ok=True)
for name in ['prepare_public_sync_inventory_20261027.py','sync_public_inventory_20261027.py']:
 shutil.copy2(T/name,M/name)
(M/'.gitattributes').write_text('*.py -text\n',encoding='utf-8')
(M/'README.md').write_text('Original location: work/textop_reproduction_20261024. These maintenance source snapshots record the public-only incremental inventory and publisher used on2026-10-27. They are not experiment commands and should not be rerun from this archived directory. No training/data deletion/large-release upload is performed.\n',encoding='utf-8')
names=['README.md','SYNC_SCOPE_AND_DATA_RECOVERY.md','scripts/audit_candidate_collision_v1.py','reports/filter_first_restored_20261027.md','results/checkpoint_sync_state.json','results/generator8278_20261027/generator8278_vae200k_train_v1/progress.json','results/teacher10000_full_filter_20261027/progress.json']
for folder in ['sources/task_helpers_20261027','sources/sync_maintenance_20261027','results/github_sync_inventory_20261027']:
 names+=sorted(p.relative_to(B).as_posix() for p in (B/folder).rglob('*') if p.is_file())
assert len(names)==len(set(names))
assert all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 and '.private.' not in n for n in names)
guard=(T/'publish.lock').open('a+b')
if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
def git(*args,stdin=None):
 return subprocess.run(['git','-C',str(R),'-c','core.quotepath=false',*args],input=stdin,capture_output=True,text=True,encoding='utf-8',check=True,timeout=60,env=env).stdout.strip()
assert git('branch','--show-current')==BRANCH and not git('diff','--cached','--name-only')
base=git('rev-parse','HEAD')
assert git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]==base, 'Remote moved; inspect before staging'
allow={'baselines/textop/'+n for n in names}
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow
print(git('commit','-m','Synchronize missing TextOp helper sources and public evidence with private data recovery inventory').splitlines()[0],flush=True)
head=git('rev-parse','HEAD')
pack=subprocess.run(['git','-C',str(R),'pack-objects','--stdout','--revs'],input=(head+'\n^'+base+'\n').encode(),capture_output=True,check=True,timeout=60)
estimated=len(pack.stdout)
assert estimated<5*1024**2, 'Unexpected large Git increment; preserve commit and inspect before upload'
print(json.dumps({'estimated_git_object_pack_bytes':estimated,'large_assets_upload':False,'changed_allowlist_files':len(names)}),flush=True)
git('push','--porcelain','origin',BRANCH)
remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==head
receipt={'branch':BRANCH,'commit':head,'verified_remote_head':remote,'url':'https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+BRANCH+'/baselines/textop'}
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
records=[]
for n in names:
 src=B/n;dst=D/n if n.startswith('results/') else D/'code/public_sync_inventory_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 h=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
 records.append({'source':n,'destination':str(dst),'sha256':h,'bytes':src.stat().st_size})
shutil.copy2(B/'SYNC_SCOPE_AND_DATA_RECOVERY.md',D/'GitHub同步范围与数据恢复索引_20261027.md')
record=dict(receipt,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,estimated_git_object_pack_bytes=estimated,actual_vpn_billed_bytes_unknown=True,large_release_assets_uploaded_bytes=0,private_arrays_or_checkpoints_published=False,unrelated_changes_staged=False,active_training_modified=False)
out=D/'GitHub同步记录_完整公开范围增量_20261027.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':head,'remote_verified':True,'D_files_sha_verified':len(records),'estimated_git_object_pack_bytes':estimated,'private_assets_uploaded':False,'record':str(out)},ensure_ascii=False),flush=True)
