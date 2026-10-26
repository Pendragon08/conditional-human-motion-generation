"""Future formal exporter supervisor; disabled until actual40k completion and review."""
import argparse,fcntl,hashlib,json,os,pathlib,re,shutil,signal,subprocess,sys,time
from apply_tracking_filter_v3 import require_completed_training
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['formal','validation'],default='formal');p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--training-receipt',required=True);p.add_argument('--training-exit-receipt',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
specpath=ROOT/'configs/contact-filter-v4-spec.json';spec=json.loads(specpath.read_text())
from contact_replay_modes import authorize,gate_mode,finish_status
gate_mode(a.mode,spec)
training=json.loads(pathlib.Path(a.training_receipt).read_text())
training_exit=json.loads(pathlib.Path(a.training_exit_receipt).read_text())
diagnostic=authorize(a.mode,spec,training,training_exit)
checkpoint=pathlib.Path(a.checkpoint);assert checkpoint.is_file() and checkpoint.stat().st_uid==1013
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==training['latest_checkpoint']['sha256']
for name,key in [('evaluate_contact_filter_v4.py','executor_sha256'),('apply_tracking_filter_v3.py','acceptor_sha256'),('contact_replay_modes.py','mode_runtime_sha256'),('contact_process_supervision.py','supervision_sha256'),('contact_filter_resources.py','resource_sampler_sha256'),('supervise_contact_filter_v4.py','supervisor_sha256')]:assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==spec[key]
out=ROOT/'tracker_validation'/a.run;proofpath=ROOT/'receipts'/('strict_contact_'+a.run+'_exit.json')
assert not out.exists() and not proofpath.exists()
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not {str(ROOT/'scripts'/x) for x in ['run_tracker.sh','train_filter_policy.py','evaluate_strict_filter.py','evaluate_strict_filter_contact.py','evaluate_contact_filter_v4.py']}.intersection(argv),'Active personal SDK worker'
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert mem>=cap['minimum_available_host_ram_gib']*1024**3 and shutil.disk_usage(ROOT).free>spec['minimum_preflight_disk_free_gib']*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()};assert free[1]>20*1024
command=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/evaluate_contact_filter_v4.py'),'--spec',str(ROOT/'configs/tracker-env-spec.json'),'--replay-spec',str(specpath),'--training-plan',str(ROOT/'configs/filter-training-plan.json'),'--mode',a.mode,'--checkpoint',str(checkpoint),'--training-receipt',a.training_receipt,'--training-exit-receipt',a.training_exit_receipt,'--output',str(out),'--device','cuda:1','--headless']
prefix='strict_contact_'+a.run
samples=ROOT/'logs'/(prefix+'_resources.jsonl')
process_proof=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(command,ROOT/'logs'/(prefix+'.log'),samples,process_proof,
 lambda pid:sample_resources(ROOT,pid,int(spec['minimum_runtime_available_host_ram_gib']*1024**3),int(spec['minimum_runtime_disk_free_gib']*1024**3)),
 spec['validation_supervisor_timeout_seconds'] if diagnostic else spec['formal_supervisor_timeout_seconds'],spec['resource_sample_interval_seconds'])
rpath=out/'receipt.json';r={}
try:
 if rpath.exists():r=json.loads(rpath.read_text())
except (OSError,ValueError) as error:proof['evaluation_receipt_error']=repr(error)
proof.update(run=a.run,requested_mode=a.mode,validation_only=diagnostic,numerical_status=r.get('status'),evaluation_receipt_sha256=hashlib.sha256(rpath.read_bytes()).hexdigest() if rpath.exists() else None,
 resources_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),process_exit_sha256=hashlib.sha256(process_proof.read_bytes()).hexdigest(),
 dataset_accepted=False,full_generator_training_started=False)
with proofpath.open('x') as stream:stream.write(json.dumps(proof,indent=2))
print(json.dumps(proof))
assert proof['supervision_passed'] and r.get('status')==finish_status(a.mode) and r.get('output_writers_closed') is True
# Deliberately separate: actual exporter exit is persisted before acceptance is possible.
