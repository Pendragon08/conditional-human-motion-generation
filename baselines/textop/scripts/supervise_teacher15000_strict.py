"""Own locked frozen15k strict replay; no training or overlapping SDK worker."""
import argparse, fcntl, hashlib, json, os, pathlib, re, shutil, subprocess
from teacher15000_strict_protocol import gate, load_teacher, authorize_teacher, verify_smoke_witnesses, STATUS
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
path=ROOT/'configs/teacher15000-direct-strict-spec.json';spec=json.loads(path.read_text());gate(a.stage,spec)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
teacher=load_teacher(ROOT,spec);checkpoint=authorize_teacher(spec,teacher)
assert sha(pathlib.Path(checkpoint['path']))==checkpoint['sha256']
if a.stage=='full':verify_smoke_witnesses(ROOT,spec,checkpoint['sha256'])
for name,value in teacher['resume_spec']['frozen_bindings'].items():assert sha(ROOT/name)==value
for name,key in [('evaluate_teacher15000_strict.py','executor_sha256'),('teacher15000_strict_protocol.py','protocol_sha256'),
                 ('direct_strict_trace.py','trace_sha256'),('supervise_teacher15000_strict.py','supervisor_sha256'),
                 ('contact_process_supervision.py','supervision_sha256'),('contact_filter_resources.py','resource_sampler_sha256'),
                 ('strict_tracking_coverage.py','coverage_sha256')]:assert sha(ROOT/'scripts'/name)==spec[key]
out=ROOT/'tracker_validation'/a.run;prefix='teacher15000_strict_'+a.run
proofpath=ROOT/'receipts'/(prefix+'_exit.json')
assert not out.exists() and not proofpath.exists()
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
blocked=['run_tracker.sh','train_filter_policy.py','evaluate_strict_filter.py','evaluate_strict_filter_contact.py','evaluate_contact_filter_v4.py','evaluate_quick_pilot.py','evaluate_teacher10000_pilot.py','evaluate_teacher15000_strict.py']
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not {str(ROOT/'scripts'/x) for x in blocked}.intersection(argv),'Existing personal SDK worker: no overlapping pilot'
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert mem>=cap['minimum_available_host_ram_gib']*1024**3
assert shutil.disk_usage(ROOT).free>spec[a.stage+'_minimum_preflight_disk_free_gib']*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()}
assert free[1]>20*1024
command=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/evaluate_teacher15000_strict.py'),
         '--pilot-spec',str(path),'--stage',a.stage,'--output',str(out),'--device','cuda:1','--headless']
samples=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
proof=run_supervised(command,ROOT/'logs'/(prefix+'.log'),samples,process,
 lambda pid:sample_resources(ROOT,pid,int(spec['minimum_runtime_available_host_ram_gib']*1024**3),int(spec['minimum_runtime_disk_free_gib']*1024**3)),
 spec[a.stage+'_timeout_seconds'],spec['resource_sample_interval_seconds'])
rpath=out/'receipt.json';r={}
try:
 if rpath.exists():r=json.loads(rpath.read_text())
except (OSError,ValueError) as error:proof['evaluation_receipt_error']=repr(error)
proof.update(run=a.run,requested_stage=a.stage,numerical_status=r.get('status'),mode='teacher15000_strict',quality_label=spec['quality_label'],
             evaluation_receipt_sha256=sha(rpath) if rpath.exists() else None,resources_sha256=sha(samples),process_exit_sha256=sha(process),
             dataset_accepted=False,physical_filter_executed=bool(a.stage=='full' and r.get('status')==STATUS),full_generator_training_started=False)
with proofpath.open('x') as stream:stream.write(json.dumps(proof,indent=2))
print(json.dumps(proof))
assert proof['supervision_passed'] and r.get('status')==STATUS and r.get('output_writers_closed') is True
