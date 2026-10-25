"""Bounded diagnostic supervisor retaining real exit/timeout and failure logs."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--test',choices=['kernel','environment'],default='kernel');a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run) and os.getuid()==1013
out=ROOT/'tracker_validation'/a.run;assert not out.exists()
entry='tracker_kernel_witness.py' if a.test=='kernel' else 'tracker_environment_probe.py'
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts'/entry),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--output',str(out),'--device','cuda:1','--headless']
if a.test=='environment':cmd+=['--reference-witness',str(ROOT/'tracker_validation/author_v5')]
log=ROOT/'logs'/('tracker_kernel_'+a.run+'.log');start=time.monotonic();timeout=False
with log.open('xb') as target:
    child=subprocess.Popen(cmd,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
        preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
    try:code=child.wait(timeout=240)
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
proof={'run':a.run,'command':cmd,'child_pid':child.pid,'child_exit_code':code,'timed_out':timeout,'elapsed_seconds':time.monotonic()-start,
       'scope':'zero-update numerical diagnostic only','full_training_started':False}
(ROOT/'receipts'/('tracker_kernel_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2))
if out.exists() and (out/'receipt.json').exists():
    receipt=json.loads((out/'receipt.json').read_text());proof['numerical_status']=receipt['status']
print(json.dumps(proof),flush=True)
expected='TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED' if a.test=='kernel' else 'TEXTOP_TRACKER_ENVIRONMENT_PROBE_PASSED'
if code==0 and not timeout and proof.get('numerical_status')==expected:
    print('TEXTOP_TRACKER_SUPERVISED_EXIT0',flush=True)
else:
    print(log.read_text(errors='replace')[-7000:]);sys.exit(1)
