"""Read only lightweight full replay metadata; never open private trace arrays."""
import argparse,json,pathlib,time
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
assert all(x.isidentifier() and x.isascii() for x in [a.run,a.evidence])
out=B/'results/teacher15000_direct_strict_20261027'/a.run;out.mkdir(parents=True,exist_ok=True)
code=r'''
import pathlib,json,hashlib,os,fcntl,subprocess,time,array,math
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;out=r/'tracker_validation'/run;prefix='teacher15000_strict_'+run
assert os.getuid()==1013
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'run':run,'observed_unix':time.time(),'files':{},'sha256':{},'workers':[]}
paths={'receipt':out/'receipt.json','driver':r/'receipts'/(prefix+'_driver.json'),'process':r/'receipts'/(prefix+'_process_exit.json'),'exit':r/'receipts'/(prefix+'_exit.json'),'outer':r/'receipts'/(prefix+'_outer_exit.json')}
for k,p in paths.items():
 if p.exists():result['files'][k]=json.loads(p.read_bytes());result['sha256'][k]=sha(p)
for k in ['.log','_outer.log']:
 p=r/'logs'/(prefix+k)
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3500));result[k+'_tail']=f.read().decode('utf-8',errors='replace')
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit() or int(q.name)==os.getpid():continue
 try:
  if q.stat().st_uid!=1013:continue
  argv=(q/'cmdline').read_bytes().decode(errors='replace').split('\0');st=(q/'stat').read_text().rsplit(')',1)[1].split()
  if str(r/'scripts/evaluate_teacher15000_strict.py') in argv or str(r/'scripts/supervise_teacher15000_strict.py') in argv:
   result['workers'].append(dict(pid=int(q.name),start_ticks=st[19],state=st[0],argv=argv))
 except (OSError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['common_lock_free']=True
except BlockingIOError:result['common_lock_free']=False
finally:lock.close()
f=result['files'];result['completed_execution_verified']=False
result['full_execution_closed']=bool('outer' in f and 'exit' in f and 'receipt' in f and f['outer']['actual_returncode']==0 and f['receipt'].get('output_writers_closed'))
result['closed_trace_cpu_audit_required']=result['full_execution_closed']
result['private_trace_files_read']=0
if result['full_execution_closed']:
 result['closed_evidence_metadata']={k:result['sha256'][k] for k in ['receipt','process','exit','outer']}
 result['next_action']='Use the new durable CPU audit before accepting physical decisions or constructing generator data'
print(json.dumps(result))
'''.replace('RUN',repr(a.run))
result=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=120))
(out/(a.evidence+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
(out/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
brief={k:result[k] for k in ['run','observed_unix','common_lock_free','completed_execution_verified']}
brief['receipt']=result['files'].get('receipt');brief['outer']=result['files'].get('outer');brief['worker_pids']=[w['pid'] for w in result['workers']]
brief['exit']=result['files'].get('exit');brief['log_tail']=result.get('.log_tail','')[-1200:];brief['outer_tail']=result.get('_outer.log_tail','')[-1800:]
print(json.dumps(brief))
