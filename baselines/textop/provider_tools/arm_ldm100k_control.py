"""Bind two real CPU witnesses, then launch one durable wait-for100k observer."""
import argparse,ast,base64,hashlib,json,pathlib,shutil
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/ldm100k_user_handoff_20261028'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--document-summary',required=True);a=p.parse_args()
docpath=pathlib.Path(a.document_summary);doc=json.loads(docpath.read_bytes());author=json.loads((O/'control_fixture_author_v1.json').read_bytes())
assert doc['status']=='LDM100K_CONTROL_FRESH_DOCUMENT_VERIFIED' and doc['actual_exit_code']==0 and doc['findings']==[]
assert doc['actual_fixture_child_reaped'] and doc['actual_fixture_child_streams_final']
assert author['actual_exit_code']==0 and author['fixture']['production_process_signals']==0
assert doc['fresh_document_fixture']['production_process_signals']==0 and doc['client']['actual_client_exit_code']==0
assert doc['client']['client_reaped'] and doc['client']['streams_final'] and not doc['client']['timed_out']
for n,h in doc['current_source_spec_sha256'].items():assert sha(B/n)==h,n
sp=B/'configs/ldm100k-user-control-spec.json';old=sp.read_bytes();s=json.loads(old);assert not s['control_authorized']
s.update(control_authorized=True,author_cpu_witness_sha256=sha(O/'control_fixture_author_v1.json'),fresh_document_summary_sha256=sha(docpath),
 fresh_remote_fixture_summary_sha256=doc['remote_fixture_summary_sha256'])
template=base64.b64encode((json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode()).decode()
code=r'''
import base64,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sp=r/'configs/ldm100k-user-control-spec.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(sp)==OLD;assert not json.loads(sp.read_bytes())['control_authorized']
s=json.loads(base64.b64decode(TEMPLATE));sys.path.insert(0,str(r/'scripts'))
from ldm100k_control_protocol import verify_identity
for n,h in s['bound_sources'].items():assert sha(r/n)==h
verify_identity(s['producer']);verify_identity(s['supervisor'])
fixture=r/'ldm100k_user_handoff_20261028/control_fixture_document_v1/SUMMARY.json'
assert sha(fixture)==s['fresh_remote_fixture_summary_sha256'] and json.loads(fixture.read_bytes())['production_process_signals']==0
actual=json.loads((r/'ldm100k_user_handoff_20261028/control_fixture_author_v1/SUMMARY.json').read_bytes());assert actual['status']=='LDM100K_CONTROL_NEW_CPU_FIXTURES_PASSED' and actual['production_process_signals']==0
run='user_ldm100k_control_v1';prefix='ldm100k_control_'+run
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json','_process_exit.json'])
assert not (r/'ldm100k_user_handoff_20261028'/run).exists()
archive=r/'receipts/ldm100k_control_spec_before_enable.json'
with archive.open('xb') as f:f.write(sp.read_bytes())
raw=base64.b64decode(TEMPLATE);tmp=sp.with_name(sp.name+'.enable.partial')
with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
tmp.replace(sp)
cmd=['python3',str(r/'scripts/supervise_ldm100k_handoff.py'),'--run',run]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
source='import json,pathlib,subprocess\nq=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":q.returncode},f)\nraise SystemExit(q.returncode)\n'
with driver.open('x') as f:f.write(source)
with (r/'logs'/(prefix+'_outer.log')).open('xb') as stream:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='DURABLE_LDM100K_OBSERVER_DISPATCHED_WAIT_FOR_REAL_STATUS',run=run,driver_pid=q.pid,new_spec_sha256=sha(sp),new_spec_base64=base64.b64encode(raw).decode(),
 production_signal_sent_by_dispatch=False,new_gpu_contexts=0,original_training_source_modified=False,observed_unix=time.time())
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump({k:v for k,v in result.items() if k!='new_spec_base64'},f,indent=2)
print(json.dumps(result))
'''
code=code.replace('OLD',repr(hashlib.sha256(old).hexdigest())).replace('TEMPLATE',repr(template));ast.parse(code)
j=json.loads(recorded_remote(code,O/'arm_control_v1_client.json',timeout=90));raw=base64.b64decode(j.pop('new_spec_base64'))
assert hashlib.sha256(raw).hexdigest()==j['new_spec_sha256'];sp.write_bytes(raw)
(O/'arm_control_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
dest=pathlib.Path('D:/具身智能/TextOp复现_20261024/configs');shutil.copy2(sp,dest/sp.name)
print(json.dumps(j))
