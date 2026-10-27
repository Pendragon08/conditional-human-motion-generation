"""Explicit aggregate/code allowlist; never stage private motion or fullstate files."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,msvcrt
TASK=Path(__file__).resolve().parent;REPO=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';BASE=REPO/'baselines/textop';D=Path('D:/具身智能/TextOp复现_20261024')
BRANCH='codex/textop-reproduction-20261024'
names=['CONTINUATION.md','README.md','SERVER_NOTES.md','GENERATOR_PAUSE_ENVIRONMENT.md','GENERATOR_PAUSE_MARKER_ENVIRONMENT.md','TEACHER10000_FILTER_RECOVERY_ENVIRONMENT.md',
 'configs/filter-first-priority-plan-20261027.json','configs/generator8278-marker-pause-spec.json','configs/generator8278-pause-spec.json','configs/teacher10000-filter-recovery-spec.json',
 'provider_tools/collect_filter_first_report.py','provider_tools/collect_generator8278_report.py','provider_tools/collect_teacher_filter_recovery.py','provider_tools/launch_teacher10000_filter_recovery.py',
 'reports/filter_first_restored_20261027.md','scripts/audit_generator8278_pause_cpu.py','scripts/closed_boundary_interrupt.py','scripts/closed_boundary_marker.py','scripts/export_teacher10000_relaxed_scope.py',
 'scripts/pause_generator8278_at_boundary.py','scripts/pause_generator8278_with_marker.py','scripts/supervise_teacher10000_filter_recovery.py','scripts/validate_closed_boundary_interrupt.py','scripts/validate_closed_boundary_marker.py']
for folder in ['results/generator_pause_20261027','results/teacher10000_full_filter_20261027']:
 names+=sorted(p.relative_to(BASE).as_posix() for p in (BASE/folder).rglob('*') if p.is_file())
assert len(names)==len(set(names)) and all((BASE/n).is_file() for n in names)
assert all((BASE/n).suffix in {'.md','.py','.json','.txt'} and '.private.' not in n and (BASE/n).stat().st_size<20*1024**2 for n in names)
assert all('checkpoint_sync_state' not in n for n in names)
guard=(TASK/'publish.lock').open('a+b')
if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
def git(*args,stdin=None):
 return subprocess.run(['git','-C',str(REPO),'-c','core.quotepath=false',*args],input=stdin,capture_output=True,text=True,encoding='utf-8',check=True,timeout=60,env=env).stdout.strip()
assert git('branch','--show-current')==BRANCH and not git('diff','--cached','--name-only')
allow={'baselines/textop/'+n for n in names}
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow
print(git('commit','-m','Preserve and pause prefilter VAE; restore physics-filter-first reproduction').splitlines()[0],flush=True)
git('push','origin',BRANCH);head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==head
receipt={'branch':BRANCH,'commit':head,'verified_remote_head':remote,'url':'https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+BRANCH+'/baselines/textop'}
(BASE/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
records=[]
for n in names:
 src=BASE/n;dst=D/n if n.startswith('results/') else D/'code/filter_first_20261027'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 h=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
 records.append({'source':n,'destination':str(dst),'sha256':h,'bytes':src.stat().st_size})
record=dict(receipt,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,private_arrays_or_checkpoints_published=False,unrelated_changes_staged=False)
out=D/'GitHub同步记录_恢复论文筛选顺序_20261027.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':head,'remote_verified':True,'D_files_sha_verified':len(records),'record':str(out)},ensure_ascii=False),flush=True)
