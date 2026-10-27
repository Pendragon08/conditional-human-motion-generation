"""Bind and enable actual-pass corpus construction after positive closed CPU proof."""
import pathlib,json,hashlib,base64
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O=B/'results/filtered15000_handoff_20261027';audit_run='full_result_audit_v1'
progress=json.loads((O/audit_run/'progress.json').read_text(encoding='utf-8'))
assert progress['kind']=='audit' and progress['closed_execution_verified'] and progress['common_lock_free'] and not progress['workers']
r=progress['files']['receipt'];e=progress['files']['exit']
assert r['status']=='CLOSED_FULL_STRICT_RESULT_CPU_AUDIT_PASSED' and r['strict_result_accepted']
assert r['source_count']==r['attempted']==8278 and r['unattempted']==0 and r['complete_passes']>0
assert e['audit_receipt_sha256']==progress['sha256']['receipt']
path=B/'configs/filtered15000-corpus-spec.json';old=path.read_bytes();s=json.loads(old);assert s['build_authorized'] is False
s.update(build_authorized=True,strict_audit_receipt=ROOT+'/filtered15000_20261027/'+audit_run+'/audit.json',
 strict_audit_receipt_sha256=progress['sha256']['receipt'],strict_audit_supervisor_sha256=progress['sha256']['exit'])
new=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,sys,os,fcntl,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/filtered15000-corpus-spec.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==OLD
data=base64.b64decode(NEW);s=json.loads(data);sys.path.insert(0,str(r/'scripts'))
from filtered15000_inputs import load_selection
audit,selected=load_selection(s)
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert len(selected)>0 and not pathlib.Path(s['candidate_view']).exists()
view_parent=pathlib.Path(s['candidate_view']).parent
assert view_parent.resolve()==view_parent and view_parent.stat().st_uid==1013
assert shutil.disk_usage(view_parent).free>s['minimum_shm_free_gib']*1024**3
assert shutil.disk_usage(r).free>2*1024**3
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0];assert mem>=16*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
sources=dict(s['bound_sources'],**{'scripts/build_filtered15000_corpus.py':s['builder_sha256'],'scripts/supervise_filtered15000_corpus.py':s['supervisor_sha256']})
for name,h in sources.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
archive=r/'receipts/filtered15000_corpus_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes())
temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='ACTUAL_PASS_CORPUS_ENABLED_NOT_DISPATCHED',strict_audit_sha256=s['strict_audit_receipt_sha256'],
 selected_sources=len(selected),failed_sources=audit['failed'],source_count=8278,old_spec_sha256=OLD,new_spec_sha256=hashlib.sha256(data).hexdigest(),
 memory_available_bytes=mem,shm_free_bytes=shutil.disk_usage(view_parent).free,new_gpu_contexts=0,new_model_loads=0,new_corpus_runs=0,
 dataset_accepted=False,generator_started=False,paper_two_stage_route_reproduced=False)
with (r/'receipts/filtered15000_corpus_activation.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD',repr(expected)).replace('NEW',repr(base64.b64encode(new).decode()))
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/'enable_corpus_v1_client.json',timeout=90));path.write_bytes(new)
(folder/'corpus_activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
