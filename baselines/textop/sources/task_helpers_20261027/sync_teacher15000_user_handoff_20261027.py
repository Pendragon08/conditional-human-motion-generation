"""Publish only the user's 15k notification and accepted-corpus continuation plan."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,msvcrt
T=Path(__file__).resolve().parent
R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation'
B=R/'baselines/textop'
D=Path('D:/具身智能/TextOp复现_20261024')
BRANCH='codex/textop-reproduction-20261024'
names=['CONTINUATION.md','README.md','configs/teacher15000-user-handoff-plan-20261027.json',
 'provider_tools/collect_filter_first_report.py','reports/teacher15000_user_handoff_20261027.md',
 'reports/filter_first_restored_20261027.md','results/teacher10000_full_filter_20261027/progress.json']
assert all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 for n in names)
guard=(T/'publish.lock').open('a+b')
if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy()
env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
def git(*args,stdin=None):
 return subprocess.run(['git','-C',str(R),'-c','core.quotepath=false',*args],input=stdin,capture_output=True,text=True,encoding='utf-8',check=True,timeout=60,env=env).stdout.strip()
assert git('branch','--show-current')==BRANCH and not git('diff','--cached','--name-only')
allow={'baselines/textop/'+n for n in names}
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow
print(git('commit','-m','Record user-requested 15k filtering notification and accepted-corpus generator continuation').splitlines()[0],flush=True)
git('push','origin',BRANCH)
head=git('rev-parse','HEAD')
remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
assert remote==head
receipt={'branch':BRANCH,'commit':head,'verified_remote_head':remote,'url':'https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+BRANCH+'/baselines/textop'}
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
records=[]
for n in names:
 src=B/n;dst=D/n if n.startswith('results/') else D/'code/teacher15000_user_handoff_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 h=hashlib.sha256(src.read_bytes()).hexdigest()
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
 records.append({'source':n,'destination':str(dst),'sha256':h,'bytes':src.stat().st_size})
shutil.copy2(B/'reports/teacher15000_user_handoff_20261027.md',D/'15000轮筛选通知与生成器自动接续_20261027.md')
record=dict(receipt,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,private_arrays_or_checkpoints_published=False,unrelated_changes_staged=False,active_training_modified=False)
out=D/'GitHub同步记录_15000轮自动验收接续_20261027.json'
out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':head,'remote_verified':True,'D_files_sha_verified':len(records),'record':str(out)},ensure_ascii=False),flush=True)
