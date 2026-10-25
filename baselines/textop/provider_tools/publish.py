import pathlib,sys,subprocess,os,json,time
TASK=pathlib.Path(__file__).resolve().parent
from session import REPO
# Both finite data workers use this publisher. Serialize complete Git operations
# so one worker cannot commit the other worker's staged intermediate changes.
if os.name=='nt':
    import msvcrt
    publish_guard=(REPO.parent.parent/'textop_reproduction_20261024/publish.lock').open('a+b')
    if publish_guard.seek(0,2)==0:publish_guard.write(b'0');publish_guard.flush()
    for _ in range(120):
        try:
            publish_guard.seek(0);msvcrt.locking(publish_guard.fileno(),msvcrt.LK_NBLCK,1);break
        except OSError:time.sleep(1)
    else:raise TimeoutError('Another publisher is still active; no Git changes made')
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897')
def git(*args):return subprocess.run(['git','-C',str(REPO),'-c','core.quotepath=false',*args],check=True,env=env,capture_output=True,text=True,encoding='utf-8').stdout.strip()
branch='codex/textop-reproduction-20261024'
if git('branch','--show-current')!=branch:
    if git('branch','--list',branch):print(git('checkout',branch))
    else:print(git('checkout','-b',branch))
print(git('add','--','baselines/textop'))
meta=json.loads((REPO/'baselines/textop/sources/official_source_receipt.json').read_text())
tracked=set(git('ls-files','-z').split('\0'))
missing=['baselines/textop/upstream/'+x['path'] for x in meta['files'] if 'baselines/textop/upstream/'+x['path'] not in tracked]
if missing:print(git('add','-f','--',*missing))
public_results=[]
for p in (REPO/'baselines/textop/results').rglob('*'):
    if '.private.' in p.name or '_private.' in p.name:continue
    if p.is_file() and (p.suffix in ['.json','.jsonl','.txt','.log','.yaml'] or p.name=='complete_records.tar.gz'):public_results.append(p.relative_to(REPO).as_posix())
if public_results:print(git('add','-f','--',*public_results))
changes=git('diff','--cached','--name-only').splitlines()
if any(not x.startswith('baselines/textop/') for x in changes):raise RuntimeError('Unrelated staged changes; refusing commit')
if any('.private.' in x or '_private.' in x for x in changes):raise RuntimeError('Licensed private asset staged; refusing publication')
if changes:
    print(git('diff','--cached','--stat')[-1200:])
    print(git('commit','-m','Audit TextOp data provenance and reconstruct paper-oriented 29DoF candidates'))
print(git('push','-u','origin',branch))
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
assert head==remote
receipt={'branch':branch,'commit':head,'verified_remote_head':remote,'url':'https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+branch+'/baselines/textop'}
(TASK/'github_sync.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
