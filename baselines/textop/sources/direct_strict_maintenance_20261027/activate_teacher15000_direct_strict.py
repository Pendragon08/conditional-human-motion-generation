"""Activate full scope only after both real frozen15k witnesses have exited."""
import pathlib,sys,json,hashlib,base64
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/teacher15000_direct_strict_20261027';P=B/'configs/teacher15000-direct-strict-spec.json';old=P.read_bytes();s=json.loads(old)
assert s['full_launch_enabled'] is False and s['smoke_witnesses']==[]
review=W/'work/textop_reproduction_20261024/teacher15000_strict_document_v1/SUMMARY.json';assert review.is_file()
for run in ['teacher15000_smoke_author_v1','teacher15000_smoke_document_v1']:
 j=json.loads((O/run/'progress.json').read_text(encoding='utf-8'))
 assert j['completed_execution_verified'] and j['common_lock_free'] and not j['workers']
 assert j['files']['outer']['actual_returncode']==0 and j['files']['receipt']['complete_trials']==8
 s['smoke_witnesses'].append(dict(run=run,receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit']))
s.update(full_launch_enabled=True,independent_document_summary_sha256=hashlib.sha256(review.read_bytes()).hexdigest())
new=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,sys,os,fcntl,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/teacher15000-direct-strict-spec.json';old=p.read_bytes();assert hashlib.sha256(old).hexdigest()==OLD
new=base64.b64decode(NEW);s=json.loads(new);sys.path.insert(0,str(r/'scripts'))
from teacher15000_strict_protocol import load_teacher,authorize_teacher,verify_smoke_witnesses,settings_signature
t=load_teacher(r,s);c=authorize_teacher(s,t);verify_smoke_witnesses(r,s,c['sha256'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(r).free>s['full_minimum_preflight_disk_free_gib']*1024**3
for n,key in [('evaluate_teacher15000_strict.py','executor_sha256'),('teacher15000_strict_protocol.py','protocol_sha256'),('direct_strict_trace.py','trace_sha256'),('supervise_teacher15000_strict.py','supervisor_sha256')]:
 assert hashlib.sha256((r/'scripts'/n).read_bytes()).hexdigest()==s[key]
archive=r/'receipts/teacher15000_direct_strict_spec_before_full.json'
with archive.open('xb') as f:f.write(old)
temp=p.with_name(p.name+'.full_enable.partial')
with temp.open('xb') as f:f.write(new);f.flush();os.fsync(f.fileno())
temp.replace(p)
proof=dict(status='FULL_SCOPE_ENABLED_AFTER_TWO_ACTUAL_SMOKES',teacher_stopped=True,common_lock_free_before_launch=True,
 old_spec_sha256=OLD,new_spec_sha256=hashlib.sha256(new).hexdigest(),settings_signature=settings_signature(s),
 smoke_witnesses=s['smoke_witnesses'],document_summary_sha256=s['independent_document_summary_sha256'],
 full_scope_started=False,source_count=8278,optimizer_updates=0,thresholds_changed=False,direct_ema_gate_bypass=True)
with (r/'receipts/teacher15000_direct_strict_full_activation.json').open('x') as f:json.dump(proof,f,indent=2)
lock.close();print(json.dumps(proof))
'''.replace('OLD',repr(expected)).replace('NEW',repr(base64.b64encode(new).decode()))
result=json.loads(recorded_remote(code,O/'activate_full_v1_client.json',timeout=90))
(O/'spec_before_full.json').write_bytes(old);P.write_bytes(new)
assert hashlib.sha256(P.read_bytes()).hexdigest()==result['new_spec_sha256']
(O/'activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
