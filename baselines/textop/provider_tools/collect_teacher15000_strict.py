"""Read actual frozen-policy replay state and preserve a unique SSH outcome."""
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
if 'outer' in f and 'exit' in f and 'receipt' in f and f['outer']['actual_returncode']==0:
 import sys;sys.path.insert(0,str(r/'scripts'))
 from teacher15000_strict_protocol import STATUS,QUALITY
 from strict_tracking_coverage import verify_trial
 rec=f['receipt'];ex=f['exit'];pr=f['process']
 assert rec['status']==STATUS and rec['quality_label']==QUALITY and rec['output_writers_closed']
 assert rec['optimizer_updates']==0 and not rec['dataset_accepted'] and rec['policy_checkpoint_completed_iterations']==15000
 assert rec['policy_checkpoint_sha256']=='66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88'
 assert ex['child_exit_code']==0 and ex['child_reaped'] and ex['supervision_passed'] and not ex['timed_out'] and ex['supervisor_error'] is None
 assert ex['evaluation_receipt_sha256']==result['sha256']['receipt'] and ex['process_exit_sha256']==result['sha256']['process']
 assert pr['child_exit_code']==0 and pr['child_reaped'] and not pr['timed_out'] and pr['sample_errors']==[] and not pr.get('cleanup_error')
 samples=r/'logs'/(prefix+'_resources.jsonl');sr=[json.loads(x) for x in samples.read_text().splitlines()]
 assert sha(samples)==ex['resources_sha256'] and len(sr)==pr['samples_written'] and sr[-1]['phase']=='after_exit'
 strict=out/'strict.private.json';journal=out/'strict_trials.private.jsonl';index=out/'executed_traces.private.json'
 assert sha(strict)==rec['private_strict_log_sha256'] and sha(journal)==rec['private_trial_journal_sha256'] and sha(index)==rec['private_executed_trace_index_sha256']
 rows=json.loads(strict.read_text())['motions'];records=json.loads(index.read_text());count=8 if rec['stage']=='smoke' else 8278
 assert len(rows)==len(records)==count==rec['finished_trials']==rec['replayed_original_sources']
 expected=[];lookup={x['motion_id']:x for x in rows};complete=0;failed=0
 for row in rows:
  assert len(row['trials'])==1;t=row['trials'][0];verify_trial(t,t['reference_frames'],0)
  expected.append(dict(motion_id=row['motion_id'],trial=t));complete+=int(t['complete']);failed+=int(t['failed'])
 enc=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 assert sorted(map(enc,expected))==sorted(map(enc,map(json.loads,journal.read_text().splitlines())))
 ids=[];total=0
 for item in records:
  folder=pathlib.Path(item['folder']);assert folder.resolve().parent==out/'execution_private'
  meta=json.loads((folder/'trace.private.json').read_text());bp=folder/'qpos.private.bin';raw=bp.read_bytes()
  assert meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
  assert meta['trial']==lookup[meta['motion_id']]['trials'][0]
  assert meta['samples']==item['samples']==meta['trial']['control_steps']+1
  assert len(raw)==meta['binary_bytes']==meta['samples']*36*4
  assert hashlib.sha256(raw).hexdigest()==meta['binary_sha256']==item['binary_sha256']
  vals=array.array('f');vals.frombytes(raw);assert all(math.isfinite(x) for x in vals)
  assert all(.99<math.sqrt(sum(v*v for v in vals[k+3:k+7]))<1.01 for k in range(0,len(vals),36))
  ids.append(meta['motion_id']);total+=len(raw)
 assert sorted(ids)==sorted(lookup) and total==rec['private_trace_bytes']
 assert complete==rec['complete_trials'] and failed==rec['failed_trials'] and complete+failed==count
 assert rec['actual_scene_physics_steps']==4*rec['actual_vectorized_control_calls']
 result.update(completed_execution_verified=True,closed_private_traces_verified=count,finite_qpos_bytes=total,
  passed_complete_trials=complete,failed_trials=failed,unattempted=8278-count,endpoint_acceptance=False,corpus_accepted=False)
print(json.dumps(result))
'''.replace('RUN',repr(a.run))
result=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=120))
(out/(a.evidence+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
(out/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
brief={k:result[k] for k in ['run','observed_unix','common_lock_free','completed_execution_verified']}
brief['receipt']=result['files'].get('receipt');brief['outer']=result['files'].get('outer');brief['worker_pids']=[w['pid'] for w in result['workers']]
brief['exit']=result['files'].get('exit');brief['log_tail']=result.get('.log_tail','')[-1200:];brief['outer_tail']=result.get('_outer.log_tail','')[-1800:]
print(json.dumps(brief))
