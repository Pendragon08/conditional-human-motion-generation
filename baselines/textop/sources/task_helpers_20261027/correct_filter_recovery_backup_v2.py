"""Resolve v1 review gaps in independent candidate files only; keep history."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys,shutil
T=Path(__file__).resolve().parent;B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
O=B/'results/filter_recovery_checkpoint_tools_20261027';sys.path.insert(0,str(B/'provider_tools'));from session import SSH
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=T/'filter_recovery_checkpoint_document_v1'
for n,target in [('REPORT.md','review_document_v1.md'),('SUMMARY.json','review_document_v1.json')]:
 shutil.copy2(review/n,O/target)
old=json.loads((O/'disabled_runtime_bytes_v1.json').read_text(encoding='utf-8'))
doc=B/'FILTER_RECOVERY_CHECKPOINT_ENVIRONMENT.md'
(O/'environment_document_v1_bytes.json').write_text(json.dumps({'sha256':sha(doc),'base64':base64.b64encode(doc.read_bytes()).decode()},indent=2),encoding='utf-8')
sp=B/'configs/filter-recovery-backup-spec.json';spec=json.loads(sp.read_text(encoding='utf-8'))
names=['scripts/inspect_filter_recovery_checkpoint.py','scripts/filter_recovery_destination.py','scripts/validate_filter_recovery_destination.py']
for n in names:spec['bound_backup_files'][n]=sha(B/n)
spec['review_corrections']=['Output ownership/non-link/persistent same-volume ancestry guard before/after audit',
 'Existing validated CPU child supervisor plus durable pre-dispatch journal and full local SSH-client outcome preservation']
sp.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8',newline='\n')
names+=['configs/filter-recovery-backup-spec.json']
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
code=r'''
import pathlib,os,json,hashlib,base64,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
active=r/'configs/teacher10000-filter-recovery-spec.json';s=json.loads(active.read_text());assert sha(active)=='f39e2103ff6e9cceb3f7442f83a5c217328cb2cb18bce9cecfabb4fd774a64b4'
bound=dict(s['frozen_bindings']);bound['scripts/supervise_teacher10000_filter_recovery.py']=s['source_sha256'];bound['configs/teacher10000-filter-recovery-spec.json']=sha(active)
assert all(sha(r/n)==h for n,h in bound.items())
payload=PAYLOAD;old=OLD
for n,b64 in payload.items():
 p=r/n;b=base64.b64decode(b64)
 if n in old:assert sha(p)==old[n]['sha256']
 else:assert not p.exists()
 if n.endswith('.py'):compile(b.decode('utf-8'),str(p),'exec')
 temp=p.with_name(p.name+'.new_candidate_v2');assert not temp.exists()
 with temp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 temp.replace(p)
cmd=['python3',str(r/'scripts/validate_filter_recovery_destination.py'),'--run','destination_author_v2']
p=subprocess.run(cmd,capture_output=True,universal_newlines=True,timeout=45,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
result={'actual_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'active_bindings_unchanged':bound,'new_files':{n:sha(r/n) for n in payload}}
(r/'filter_recovery_checkpoint_tools_20261027/correction_v2_execution.json').write_text(json.dumps(result,indent=2))
assert all(sha(r/n)==h for n,h in bound.items());assert p.returncode==0,(p.stdout,p.stderr)
print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload)).replace('OLD',repr(old))
compile(code,'remote_correction','exec')
p=subprocess.run(SSH+['python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=90)
execution={'actual_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(O/'correction_v2_execution.json').write_text(json.dumps(execution,indent=2),encoding='utf-8')
assert p.returncode==0,(p.stdout,p.stderr)
j=json.loads(p.stdout);(O/'correction_v2.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
allnames=list(spec['bound_backup_files'])
allnames=[n for n in allnames if n.startswith('scripts/') and n!='scripts/inspect_formal_filter_checkpoint_v2.py']+['configs/filter-recovery-backup-spec.json','provider_tools/backup_filter_recovery_checkpoint.py','provider_tools/validate_filter_recovery_transport.py']
bundle={n:{'sha256':sha(B/n),'base64':base64.b64encode((B/n).read_bytes()).decode()} for n in allnames}
(O/'disabled_runtime_bytes_v2.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
print(json.dumps({'checks':json.loads(j['stdout'])['checks'],'exit':j['actual_returncode'],'active_files_unchanged':len(j['active_bindings_unchanged']),'backup_enabled':spec['backup_enabled']}))
