"""Sequential URDF FK then physical trial, two separately reaped SDK processes."""
import argparse,fcntl,json,os,pathlib,re,shutil,subprocess
from generated_pipeline_protocol import gate,sha
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert re.fullmatch('[a-z0-9_]+',a.run) and os.getuid()==1013
sp=ROOT/'configs/quick-generated-sdk-spec.json';spec=json.loads(sp.read_text());gate(ROOT,spec);assert sha(__file__)==spec['supervisor_sha256']
lk=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text());assert all(sha(ROOT/n)==h for n,h in dict(plan['frozen_runtime_files'],**plan['control_sources']).items())
assert not(ROOT/'generated_execution_private'/a.run).exists()
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text());mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert mem>=cap['minimum_available_host_ram_gib']*1024**3 and shutil.disk_usage(ROOT).free>2.5*1024**3
blocked=['run_tracker.sh','train_filter_policy.py','evaluate_quick_pilot.py','export_tracker_references.py','run_generated_pipeline_sdk.py']
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not {str(ROOT/'scripts'/n) for n in blocked}.intersection(argv),'Existing personal SDK worker; no overlapping dispatch'
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
phases=[]
for stage in ['export','replay']:
 prefix='generated_pipeline_'+a.run+'_'+stage;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists()
 cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/run_generated_pipeline_sdk.py'),'--stage',stage,'--run',a.run,'--device','cuda:1','--headless']
 samples=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
 proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),samples,process,lambda pid:sample_resources(ROOT,pid,2*1024**3,2*1024**3),900,15)
 rp=ROOT/'generated_execution_private'/a.run/stage/'receipt.json';r=json.loads(rp.read_text()) if rp.exists() else {}
 proof.update(stage=stage,evaluation_receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),resources_sha256=sha(samples),optimizer_updates=0)
 with ep.open('x') as f:json.dump(proof,f,indent=2)
 expected='GENERATED_REFERENCE_TRACKER_URDF_FK_EXPORTED' if stage=='export' else 'GENERATED_REFERENCE_PHYSICS_TRIAL_FINISHED'
 assert proof['supervision_passed'] and r.get('status')==expected and r.get('output_writers_closed'),(stage,proof,r)
 phases.append({'stage':stage,'receipt_sha256':sha(rp),'exit_sha256':sha(ep),'trial':r.get('trial')})
final={'status':'GENERATED_REFERENCE_PIPELINE_TRIAL_CLOSED','quality_label':'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE','run':a.run,'spec_sha256':sha(sp),'phases':phases,
 'output_writers_closed':True,'actual_sdk_processes':2,'optimizer_updates':0,'tracking_quality_accepted':False,'real_time_interactive_closed_loop_verified':False}
fp=ROOT/'generated_execution_private'/a.run/'pipeline.json';fp.write_text(json.dumps(final,indent=2));print(json.dumps(final))
