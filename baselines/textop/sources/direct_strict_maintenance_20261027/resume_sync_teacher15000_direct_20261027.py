"""One explicit incremental publication of source and aggregate15k evidence."""
from pathlib import Path
import json,hashlib,sys,os,msvcrt,shutil,datetime,subprocess
T=Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';B=R/'baselines/textop';D=Path('D:/具身智能/TextOp复现_20261024')
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_client
E=T/'teacher15000_direct_public_sync_resume_v1';E.mkdir(exist_ok=False)
M=B/'sources/direct_strict_maintenance_20261027';shutil.copy2(Path(__file__),M/Path(__file__).name)
names=['.gitattributes','README.md','CONTINUATION.md','SERVER_NOTES.md','SYNC_SCOPE_AND_DATA_RECOVERY.md',
 'TEACHER15000_DIRECT_CONTROL_ENVIRONMENT.md','TEACHER15000_DIRECT_STRICT_ENVIRONMENT.md',
 'scripts/teacher15000_evaluation_control.py','scripts/direct_strict_trace.py','scripts/teacher15000_strict_protocol.py','scripts/evaluate_teacher15000_strict.py','scripts/supervise_teacher15000_strict.py',
 'configs/teacher15000-direct-evaluation-spec.json','configs/teacher15000-direct-strict-spec.json','configs/teacher15000-direct-strict-priority-20261027.json',
 'provider_tools/teacher15000_control.py','provider_tools/analyze_teacher10000_to15000.py','provider_tools/finalize_teacher15000_report.py','provider_tools/collect_filter_first_report.py',
 'provider_tools/launch_teacher15000_strict.py','provider_tools/collect_teacher15000_strict.py','provider_tools/report_teacher15000_direct.py',
 'reports/teacher15000_filter_result_20261027.md','reports/teacher15000_direct_strict_20261027.md','reports/filter_first_restored_20261027.md',
 'results/checkpoint_sync_state.json','results/teacher10000_full_filter_20261027/progress.json']
prefix='results/filter_recovery_checkpoints_20261027/round_15000/'
names+=[prefix+x for x in ['backup.json','cpu_audit.json','relaxed_summary.json','saved_counter_comparison.json','user_result.json']]
prefix='results/teacher15000_direct_strict_20261027/'
names+=[prefix+x for x in ['current.json','activation.json','spec_before_full.json','deploy_v1.json','compact_trace_cpu_v1/proof.json','control/user_stop_v1_result.json','control/author_v1_result.json','control/document_v1_result.json','control_document/REPORT.md','control_document/SUMMARY.json','strict_document/REPORT.md','strict_document/SUMMARY.json']]
for run in ['teacher15000_smoke_author_v1','teacher15000_smoke_document_v1','teacher15000_direct_full_v1']:
 names+=[prefix+run+'/'+x for x in ['launch.json','progress.json']]
names+=sorted(p.relative_to(B).as_posix() for p in M.rglob('*') if p.is_file())
assert len(names)==len(set(names))
assert all((B/n).is_file() and (B/n).stat().st_size<2*1024**2 and '.private.' not in n for n in names)
lock=(T/'publish.lock').open('a+b')
if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
env=os.environ.copy();env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GIT_TERMINAL_PROMPT='0')
seq=[0]
def git(*args,stdin=''):
 seq[0]+=1;v=recorded_client(stdin,E/('git_'+str(seq[0])+'.json'),timeout=90,command=['git','-C',str(R),'-c','core.quotepath=false',*args],env=env)
 assert v['actual_client_exit_code']==0 and v['client_reaped'] and not v['timed_out'] and v['error'] is None,(args,v['stderr'])
 return v['stdout'].strip()
branch='codex/textop-reproduction-20261024'
assert git('branch','--show-current')==branch
base=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==base
allow={'baselines/textop/'+n for n in names}
assert set(git('diff','--cached','--name-only').splitlines())<=allow, 'Unexpected staged changes'
git('--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul',stdin='\0'.join(sorted(allow))+'\0')
assert set(git('diff','--cached','--name-only').splitlines())<=allow
git('-c','core.whitespace=cr-at-eol','diff','--cached','--check')
print(git('commit','-m','Preserve verified15k result and start user-directed full physical replay after stopping teacher').splitlines()[0],flush=True)
head=git('rev-parse','HEAD')
pack=subprocess.run(['git','-C',str(R),'pack-objects','--stdout','--revs'],input=(head+'\n^'+base+'\n').encode(),capture_output=True,check=True,timeout=60)
estimated=len(pack.stdout);assert estimated<5*1024**2
# Verify the committed bytes of the new bound sources, not only working files.
for n in ['scripts/teacher15000_evaluation_control.py','scripts/evaluate_teacher15000_strict.py','scripts/teacher15000_strict_protocol.py','scripts/direct_strict_trace.py','scripts/supervise_teacher15000_strict.py','configs/teacher15000-direct-evaluation-spec.json','configs/teacher15000-direct-strict-spec.json']:
 blob=subprocess.run(['git','-C',str(R),'show',head+':baselines/textop/'+n],capture_output=True,check=True,timeout=30).stdout
 assert blob==(B/n).read_bytes(),n
print(json.dumps({'estimated_increment_bytes':estimated,'files':len(names),'bound_source_git_bytes_exact':True}),flush=True)
git('push','--porcelain','origin',branch);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==head
receipt=dict(branch=branch,commit=head,verified_remote_head=remote,url='https://github.com/Pendragon08/conditional-human-motion-generation/tree/'+branch+'/baselines/textop')
(B/'provider_tools/github_sync.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
records=[]
for n in names:
 src=B/n;dst=D/n if n.startswith('results/') else D/'code/teacher15000_direct_20261027'/n
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 h=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
 records.append(dict(source=n,destination=str(dst),bytes=src.stat().st_size,sha256=h))
result=dict(receipt,files=records,estimated_git_object_pack_bytes=estimated,actual_vpn_billed_bytes_unknown=True,
 private_assets_published=False,release_reupload=False,active_runtime_modified=False,
 sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
dest=D/'GitHub同步记录_15000轮与全库直接物理筛查_20261027.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(E/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(commit=head,remote_verified=True,D_files_sha_verified=len(records),estimated_git_object_pack_bytes=estimated,record=str(dest)),ensure_ascii=False))
