"""Enable only the reviewed future CPU checkpoint workflow; no model dispatch."""
from pathlib import Path
import base64,hashlib,json,shutil,subprocess,sys
T=Path(__file__).resolve().parent;B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop';D=Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filter_recovery_checkpoint_tools_20261027';sys.path.insert(0,str(B/'provider_tools'));from session import SSH
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=T/'filter_recovery_checkpoint_document_v4';summary=json.loads((src/'SUMMARY.json').read_text(encoding='utf-8'))
assert summary['preparation_review_passed'] and summary['findings']==[] and summary['previous_reviewed_gates_regressed'] is False
assert summary['documented_commands_executed_once_each'] and summary['provenance']['all_local_server_bindings_match']
assert summary['positive_checkpoint_audit_or_backup_validated'] is False
for n,target in [('REPORT.md','review_document_v4.md'),('SUMMARY.json','review_document_v4.json')]:shutil.copy2(src/n,O/target)
dst=D/'diagnostics/filter_recovery_checkpoint_document_v4';assert not dst.exists();dst.mkdir()
records=[]
for p in src.rglob('*'):
 if p.is_file():
  q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
  records.append({'path':str(p.relative_to(src)),'bytes':p.stat().st_size,'sha256':sha(p)})
(O/'review_v4_private_copy.json').write_text(json.dumps({'files':records,'bytes':sum(x['bytes'] for x in records)},indent=2),encoding='utf-8')
sp=B/'configs/filter-recovery-backup-spec.json';s=json.loads(sp.read_text(encoding='utf-8'));old=sha(sp)
assert old==summary['provenance']['backup_spec_bytes_sha256'] and s['backup_enabled'] is False
assert all(sha(B/n)==h for n,h in s['bound_local_files'].items())
s['backup_enabled']=True
s['preparation_document_witness']={'path':'receipts/filter_recovery_checkpoint_document_v4.json','sha256':sha(src/'SUMMARY.json'),'scope':'Metadata, filesystem/client failure protection and static preparation review only; actual new checkpoint pending'}
sp.write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8',newline='\n')
code=r'''
import pathlib,json,hashlib,base64,os,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sp=r/'configs/filter-recovery-backup-spec.json';assert sha(sp)==OLD
active=r/'configs/teacher10000-filter-recovery-spec.json';a=json.loads(active.read_text());bound=dict(a['frozen_bindings']);bound['scripts/supervise_teacher10000_filter_recovery.py']=a['source_sha256'];bound['configs/teacher10000-filter-recovery-spec.json']='f39e2103ff6e9cceb3f7442f83a5c217328cb2cb18bce9cecfabb4fd774a64b4'
assert all(sha(r/n)==h for n,h in bound.items())
w=r/'receipts/filter_recovery_checkpoint_document_v4.json';assert not w.exists()
with w.open('xb') as f:f.write(base64.b64decode(WITNESS));f.flush();os.fsync(f.fileno())
raw=base64.b64decode(NEW);s=json.loads(raw);old=json.loads(sp.read_text());comparison=dict(s);comparison.pop('preparation_document_witness');comparison['backup_enabled']=False;assert comparison==old
q=sp.with_name(sp.name+'.enable_v4');assert not q.exists()
with q.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
q.replace(sp)
sys.path.insert(0,str(r/'scripts'));from filter_recovery_backup_protocol import preflight
pending=preflight(r,s,15000);assert pending['ready'] is False and pending['checkpoint_loads']==0
assert all(sha(r/n)==h for n,h in bound.items())
assert sha(w)==s['preparation_document_witness']['sha256']
print(json.dumps({'status':'FUTURE_RECOVERY_CHECKPOINT_WORKFLOW_ENABLED_REAL_CHECKPOINT_PENDING','spec_sha256':sha(sp),'backup_enabled':True,'pending':pending,'active_bindings_unchanged':bound,'preparation_document_sha256':sha(w),'positive_checkpoint_audit_or_backup_validated':False,'strict_physical_filter_completed':False,'sdk_runs':0,'training_signals':0}))
'''.replace('OLD',repr(old)).replace('WITNESS',repr(base64.b64encode((src/'SUMMARY.json').read_bytes()).decode())).replace('NEW',repr(base64.b64encode(sp.read_bytes()).decode()))
compile(code,'activation','exec');p=subprocess.run(SSH+['python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=60)
(O/'activation_execution.json').write_text(json.dumps({'actual_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2),encoding='utf-8');assert p.returncode==0,(p.stdout,p.stderr)
j=json.loads(p.stdout);(O/'activation.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
bundle=json.loads((O/'disabled_runtime_bytes_v4.json').read_text(encoding='utf-8'))
bundle['configs/filter-recovery-backup-spec.json']={'sha256':sha(sp),'base64':base64.b64encode(sp.read_bytes()).decode()}
(O/'enabled_runtime_bytes_v4.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
print(json.dumps({k:j[k] for k in ('status','spec_sha256','backup_enabled','positive_checkpoint_audit_or_backup_validated')}))
