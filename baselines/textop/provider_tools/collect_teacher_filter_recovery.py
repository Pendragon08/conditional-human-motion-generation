"""Read-only production continuation progress; no launch, signal or checkpoint copy."""
import json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
code='''import hashlib,json,pathlib,re,time,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sp=r/'configs/teacher10000-filter-recovery-spec.json';s=json.loads(sp.read_text());run=s['run'];prefix='filter_recovery_'+run
out=r/'tracker_validation'/run;rp=out/'receipt.json';receipt=json.loads(rp.read_text()) if rp.exists() else None
log=r/'logs'/(prefix+'.log');tail=''
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
tail=re.sub(r'\\x1b\\[[0-9;]*m','',tail)
iterations=[int(x) for x in re.findall(r'Learning iteration\\s+(\\d+)\\s*/',tail)]
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=1013:continue
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  if str(out) in args or str(r/'receipts'/(prefix+'_driver.py')) in args or str(r/'scripts/supervise_teacher10000_filter_recovery.py') in args:
   workers.append({'pid':int(p.name),'args':args,'cpu_affinity':sorted(os.sched_getaffinity(int(p.name))),
     'state':[x for x in (p/'status').read_text().splitlines() if x.startswith(('State:','VmRSS:'))]})
 except (OSError,ProcessLookupError):pass
ex=r/'receipts'/(prefix+'_exit.json');outer=r/'receipts'/(prefix+'_outer_exit.json');live=r/'receipts'/(prefix+'_live.json')
j={'run':run,'sample_unix':time.time(),'spec_sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),
 'receipt':receipt,'latest_native_logged_iteration_zero_based':max(iterations) if iterations else None,
 'observed_completed_from_native_log':max(iterations)+1 if iterations else None,
 'exact_source_resume_completed':10000,'exact_source_resume_updates':200000,
 'actual_workers':workers,'exit':json.loads(ex.read_text()) if ex.exists() else None,
 'outer_exit':json.loads(outer.read_text()) if outer.exists() else None,
 'latest_resource_sample':json.loads(live.read_text()) if live.exists() else None,
 'full_generator_training_started':False,'paper_quality_accepted':False}
print(json.dumps(j))
'''
j=json.loads(remote(code,timeout=60));o=B/'results/teacher10000_full_filter_20261027'
(o/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps({'run':j['run'],'receipt_status':(j['receipt'] or {}).get('status'),
 'resume_exact_state_verified':(j['receipt'] or {}).get('resume_exact_state_verified'),
 'completed_receipt':(j['receipt'] or {}).get('completed_learning_iterations'),'observed_completed_log':j['observed_completed_from_native_log'],
 'workers':[x['pid'] for x in j['actual_workers']],'outer_exit':j['outer_exit']}))
