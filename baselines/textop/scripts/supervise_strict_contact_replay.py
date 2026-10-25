"""Future formal exporter supervisor; disabled until actual40k completion and review."""
import argparse,fcntl,hashlib,json,os,pathlib,re,shutil,signal,subprocess,sys,time
from apply_tracking_filter_v3 import require_completed_training
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--training-receipt',required=True);p.add_argument('--training-exit-receipt',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
specpath=ROOT/'configs/strict-contact-replay-spec.json';spec=json.loads(specpath.read_text())
assert spec['formal_evaluation_enabled'] is True,'Future exporter disabled; no SDK dispatched'
assert spec['real_sdk_validation_completed'] is True,'Independent real SDK validation and resource review still required'
training=json.loads(pathlib.Path(a.training_receipt).read_text());require_completed_training(training)
training_exit=json.loads(pathlib.Path(a.training_exit_receipt).read_text())
assert training_exit['child_exit_code']==0 and not training_exit['timed_out'] and training_exit['numerical_status']==training['status']
checkpoint=pathlib.Path(a.checkpoint);assert checkpoint.is_file() and checkpoint.stat().st_uid==1013
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==training['latest_checkpoint']['sha256']
for name,key in [('evaluate_strict_filter_contact.py','executor_sha256'),('apply_tracking_filter_v3.py','acceptor_sha256')]:assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==spec[key]
out=ROOT/'tracker_validation'/a.run;proofpath=ROOT/'receipts'/('strict_contact_'+a.run+'_exit.json')
assert not out.exists() and not proofpath.exists()
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not {str(ROOT/'scripts'/x) for x in ['run_tracker.sh','train_filter_policy.py','evaluate_strict_filter.py','evaluate_strict_filter_contact.py']}.intersection(argv),'Active personal SDK worker'
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert mem>=cap['minimum_available_host_ram_gib']*1024**3 and shutil.disk_usage(ROOT).free>2*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()};assert free[1]>20*1024
command=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/evaluate_strict_filter_contact.py'),'--spec',str(ROOT/'configs/tracker-env-spec.json'),'--replay-spec',str(specpath),'--training-plan',str(ROOT/'configs/filter-training-plan.json'),'--mode','formal','--checkpoint',str(checkpoint),'--training-receipt',a.training_receipt,'--training-exit-receipt',a.training_exit_receipt,'--output',str(out),'--device','cuda:1','--headless']
start=time.monotonic();timed_out=False
with (ROOT/'logs'/('strict_contact_'+a.run+'.log')).open('xb') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{0,1,2,3}))
 try:code=child.wait(timeout=spec['formal_supervisor_timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
rpath=out/'receipt.json';r=json.loads(rpath.read_text()) if rpath.exists() else {}
proof={'run':a.run,'child_pid':child.pid,'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,'command':command,'numerical_status':r.get('status'),'evaluation_receipt_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest() if rpath.exists() else None,'dataset_accepted':False,'full_generator_training_started':False}
proofpath.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
assert code==0 and not timed_out and r.get('status')=='TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED' and r.get('output_writers_closed') is True
# Deliberately separate: actual exporter exit is persisted before acceptance is possible.
