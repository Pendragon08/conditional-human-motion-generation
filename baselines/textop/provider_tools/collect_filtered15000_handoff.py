"""Read post-filter job receipts/exits; no data arrays, model load or signal."""
import argparse,pathlib,json
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['audit','corpus','vae'],required=True);p.add_argument('--run',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
assert all(x.isascii() and x.isidentifier() for x in [a.run,a.evidence])
code=r'''
import pathlib,json,hashlib,os,fcntl,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
kind=KIND;run=RUN;base=r/'filtered15000_20261027'
prefix={'audit':'teacher15000_full_audit_','corpus':'filtered15000_corpus_','vae':'filtered15000_vae_'}[kind]+run
rp=base/run/'audit.json' if kind=='audit' else base/('corpus_runs' if kind=='corpus' else 'vae_runs')/run/'receipt.json'
paths=dict(receipt=rp,exit=r/'receipts'/(prefix+'_exit.json'),process=r/'receipts'/(prefix+'_process_exit.json'),outer=r/'receipts'/(prefix+'_outer_exit.json'),driver=r/'receipts'/(prefix+'_driver.json'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(kind=kind,run=run,observed_unix=time.time(),files={},sha256={},workers=[],closed_execution_verified=False,arrays_read=0,model_loads=0)
for key,p in paths.items():
 if p.is_file():result['files'][key]=json.loads(p.read_bytes());result['sha256'][key]=sha(p)
for key in ['.log','_outer.log']:
 p=r/'logs'/(prefix+key)
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));result[key+'_tail']=f.read().decode('utf-8',errors='replace')
names={'audit':['audit_teacher15000_full_result.py','supervise_teacher15000_full_audit.py'],
 'corpus':['build_filtered15000_corpus.py','supervise_filtered15000_corpus.py'],
 'vae':['run_filtered15000_vae.py','supervise_filtered15000_vae.py']}[kind]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=1013:continue
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');st=(p/'stat').read_text().rsplit(')',1)[1].split()
  if any(str(r/'scripts'/n) in argv for n in names):result['workers'].append(dict(pid=int(p.name),start_ticks=st[19],state=st[0],argv=argv))
 except (OSError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['common_lock_free']=True
except BlockingIOError:result['common_lock_free']=False
finally:lock.close()
f=result['files']
if all(k in f for k in ['receipt','exit','process','outer']) and f['outer']['actual_returncode']==0:
 rec=f['receipt'];ex=f['exit'];pr=f['process']
 expected={'audit':['CLOSED_FULL_STRICT_RESULT_CPU_AUDIT_PASSED'],'corpus':['FILTERED15000_CORPUS_LOADER_WITNESS_PASSED'],
  'vae':['FILTERED15000_VAE_WITNESS_PASSED','FILTERED15000_VAE_200000_TRAINING_FINISHED']}[kind]
 assert rec['status'] in expected and rec['output_writers_closed']
 for e in [ex,pr]:assert e['supervision_passed'] and e['child_reaped'] and e['child_exit_code']==0 and not e['timed_out'] and e['supervisor_error'] is None and e['sample_errors']==[] and not e.get('cleanup_error')
 key='audit_receipt_sha256' if kind=='audit' else 'receipt_sha256' if kind=='corpus' else 'evaluation_receipt_sha256'
 assert ex[key]==result['sha256']['receipt'] and ex['process_exit_sha256']==result['sha256']['process']
 samples=r/'logs'/(prefix+'_resources.jsonl');rs=[json.loads(x) for x in samples.read_text().splitlines()]
 sk='resource_sha256' if kind=='corpus' else 'resources_sha256';assert sha(samples)==ex[sk]
 assert len(rs)==pr['samples_written']==ex['samples_written'] and rs[-1]['phase']=='after_exit'
 assert not result['workers'];result['closed_execution_verified']=True
print(json.dumps(result))
'''.replace('KIND',repr(a.kind)).replace('RUN',repr(a.run))
out=B/'results/filtered15000_handoff_20261027'/a.run;out.mkdir(parents=True,exist_ok=True)
j=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=90));(out/(a.evidence+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8');(out/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
