"""One durable post-filter job; disabled specs and missing real data fail closed."""
import argparse,pathlib,json,hashlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','build','verify','recovery','accept_data','vae_witness','vae_train'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
if a.stage=='audit':specname='teacher15000-full-result-audit-spec.json';script='supervise_teacher15000_full_audit.py';flags=[];prefix='teacher15000_full_audit_'
elif a.stage in ['build','verify']:specname='filtered15000-corpus-spec.json';script='supervise_filtered15000_corpus.py';flags=['--mode',a.stage];prefix='filtered15000_corpus_'
elif a.stage=='recovery':specname='filtered15000-recovery-spec.json';script='supervise_filtered15000_recovery.py';flags=[];prefix='filtered15000_recovery_'
elif a.stage=='accept_data':specname='filtered15000-recovery-spec.json';script='supervise_filtered15000_acceptance.py';flags=[];prefix='filtered15000_acceptance_'
else:specname='filtered15000-vae-spec.json';script='supervise_filtered15000_vae.py';flags=['--mode','witness' if a.stage=='vae_witness' else 'train'];prefix='filtered15000_vae_'
expected=hashlib.sha256((B/'configs'/specname).read_bytes()).hexdigest()
code=r'''
import pathlib,json,hashlib,subprocess,os,fcntl,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
stage=STAGE;run=RUN;name=SPEC;prefix=PREFIX+run;sp=r/'configs'/name
assert hashlib.sha256(sp.read_bytes()).hexdigest()==EXPECTED;s=json.loads(sp.read_text())
sys.path.insert(0,str(r/'scripts'))
if stage=='audit':
 assert s['launch_enabled'] is True
 from audit_teacher15000_full_result import load_closed_run
 load_closed_run()
elif stage in ['build','verify']:
 assert s['build_authorized'] is True
 from filtered15000_inputs import load_selection
 load_selection(s)
elif stage=='recovery':
 assert s['prepare_authorized'] is True and s['corpus_spec_sha256'] and len(s['corpus_witnesses'])==2 and s['fresh_document_summary_sha256']
 from filtered15000_inputs import load_selection
 load_selection(json.loads((r/'configs/filtered15000-corpus-spec.json').read_text()))
elif stage=='accept_data':
 assert s['prepare_authorized'] and s['acceptance_authorized']
 assert (r/'receipts/filtered15000_corpus_recovery_private_backup.json').is_file()
else:
 assert s['witness_authorized'] and (stage=='vae_witness' or s['train_authorized'])
 assert s['data_acceptance_sha256'] and s['corpus_receipt_sha256'] and s['strict_audit_receipt_sha256']
 from filtered15000_inputs import load_selection
 load_selection(json.loads((r/'configs/filtered15000-corpus-spec.json').read_text()))
paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_outer_exit.json','_process_exit.json']]
assert not any(p.exists() for p in paths),'Existing job intent/outcome: inspect, never retry blindly'
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
cmd=['python3',str(r/'scripts'/SCRIPT)]+FLAGS+['--run',run]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,json,pathlib\nq=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':q.returncode},f)\nraise SystemExit(q.returncode)\n"
with driver.open('x') as f:f.write(text)
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='DURABLE_POST_FILTER_JOB_DISPATCHED_NOT_YET_ACCEPTED',run=run,stage=stage,driver_pid=q.pid,spec_sha256=EXPECTED,command=cmd)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
for k,v in [('STAGE',a.stage),('RUN',a.run),('SPEC',specname),('PREFIX',prefix),('EXPECTED',expected),('SCRIPT',script),('FLAGS',flags)]:code=code.replace(k,repr(v))
out=B/'results/filtered15000_handoff_20261027'/a.run;out.mkdir(parents=True,exist_ok=False)
j=json.loads(recorded_remote(code,out/'dispatch.json',timeout=90));(out/'launch.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))
