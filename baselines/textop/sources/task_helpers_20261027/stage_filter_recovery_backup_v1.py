"""Deploy NEW CPU-only checkpoint tooling, preserving all active source bytes."""
from pathlib import Path
import base64,hashlib,json,os,subprocess,sys
TASK=Path(__file__).resolve().parent; B=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from session import SSH
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O=B/'results/filter_recovery_checkpoint_tools_20261027'; O.mkdir(exist_ok=True)
names=['scripts/filter_recovery_backup_protocol.py','scripts/filter_recovery_statistics.py','scripts/inspect_filter_recovery_checkpoint.py',
 'scripts/validate_filter_recovery_backup.py','configs/filter-recovery-backup-spec.json']
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
for n in names:
 if n.endswith('.py'):compile((B/n).read_text(encoding='utf-8'),n,'exec')
code=r'''
import pathlib,os,json,hashlib,base64,subprocess,time
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sp=r/'configs/teacher10000-filter-recovery-spec.json';s=json.loads(sp.read_text());assert sha(sp)=='f39e2103ff6e9cceb3f7442f83a5c217328cb2cb18bce9cecfabb4fd774a64b4'
bindings=dict(s['frozen_bindings']);bindings['scripts/supervise_teacher10000_filter_recovery.py']=s['source_sha256'];bindings['configs/teacher10000-filter-recovery-spec.json']=sha(sp)
before={n:sha(r/n) for n in bindings};assert before==bindings
resource={'meminfo':{x.split(':')[0]:x.split(':')[1].strip() for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))},'data_free_bytes':os.statvfs(r).f_bavail*os.statvfs(r).f_frsize,'home_free_bytes':os.statvfs('/home/ouyanghao').f_bavail*os.statvfs('/home/ouyanghao').f_frsize,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],universal_newlines=True)}
payload=PAYLOAD_LITERAL
assert all(not (r/n).exists() for n in payload),'Never overwrite an existing deployment'
for n,data in payload.items():
 b=base64.b64decode(data);p=r/n
 if n.endswith('.py'):compile(b.decode('utf-8'),str(p),'exec')
 with p.open('xb') as f:f.write(b)
out=r/'filter_recovery_checkpoint_tools_20261027';out.mkdir(exist_ok=False)
cmd=['python3',str(r/'scripts/validate_filter_recovery_backup.py'),'--output',str(out/'author_v1.json')]
proc=subprocess.run(cmd,capture_output=True,universal_newlines=True,timeout=45,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
execution={'command':cmd,'actual_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
(out/'author_execution_v1.json').write_text(json.dumps(execution,indent=2))
after={n:sha(r/n) for n in bindings};assert after==before
assert proc.returncode==0,(proc.stdout,proc.stderr)
pre=subprocess.run(['python3',str(r/'scripts/inspect_filter_recovery_checkpoint.py'),'--completed','15000','--preflight-only'],capture_output=True,universal_newlines=True,timeout=45,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
pending_execution={'actual_returncode':pre.returncode,'stdout':pre.stdout,'stderr':pre.stderr}
(out/'pending_execution_v1.json').write_text(json.dumps(pending_execution,indent=2))
assert pre.returncode==0,(pre.stdout,pre.stderr)
pending=json.loads(pre.stdout);assert pending['ready'] is False and pending['checkpoint_loads']==0
assert not (r/'checkpoint_backups/round_15000/filter_recovery_author_v1').exists()
assert all(sha(r/n)==h for n,h in bindings.items())
print(json.dumps({'status':'NEW_CPU_TOOLING_DEPLOYED_METADATA_ONLY','execution':execution,'checks':json.loads(proc.stdout),'pending_execution':pending_execution,'pending':pending,'active_bindings_unchanged':before,'resource_preflight':resource,'new_files':{n:sha(r/n) for n in payload},'new_sdk_runs':0,'training_signals':0,'real_new_checkpoint_loads':0}))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('PAYLOAD_LITERAL',repr(payload))
compile(code,'remote_stage','exec')
p=subprocess.run(SSH+['python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=150)
receipt={'actual_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(O/'stage_execution_v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
assert p.returncode==0,(p.stdout,p.stderr)
j=json.loads(p.stdout);(O/'stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
bundle={n:{'sha256':hashlib.sha256((B/n).read_bytes()).hexdigest(),'base64':base64.b64encode((B/n).read_bytes()).decode()} for n in names+['provider_tools/backup_filter_recovery_checkpoint.py']}
(O/'disabled_runtime_bytes_v1.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
print(json.dumps({'status':j['status'],'checks':j['checks']['checks'],'pending':j['pending']['status'],'active_files_unchanged':len(j['active_bindings_unchanged'])}))
