"""Finite, task-local FK export or bounded PPO diagnostic; no full training."""
import argparse,json,os,pathlib,re,signal,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RAM=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--job',required=True,choices=['export_witness','export643','export1653','ppo_witness']);a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run) and os.getuid()==1013
out=(RAM/'tracker_references'/a.run) if a.job.startswith('export') else (ROOT/'tracker_validation'/a.run)
assert not out.exists(),'Immutable run already exists; use a new run name'
cmd=['bash',str(ROOT/'scripts/run_tracker.sh')]
if a.job.startswith('export'):
    expected='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED';entry='export_tracker_references.py'
    if a.job=='export1653':
        folder=RAM/'local1653_v1/prepared1653_v1'
        progress=json.loads((folder/'progress.json').read_text())
        assert progress['status']=='PREPROCESSING_FINISHED_FILTER_PENDING' and progress['completed_sequences']==1653 and progress['failed_sequences']==0
        assert __import__('shutil').disk_usage(RAM).free>12*1024**3,'Leave shared-RAM reserve; source outputs retained'
        extra=['--manifest',str(folder/'prepared_manifest.private.jsonl')]
    else:
        extra=['--manifest',str(ROOT/'data_preparation_20261025/available143_v1/prepared_manifest.private.jsonl')]
        if a.job=='export_witness':extra+=['--limit','2']
        else:extra+=['--manifest',str(ROOT/'data_preparation_20261025/available500_v1/prepared_manifest.private.jsonl')]
else:
    expected='TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED';entry='tracker_ppo_resume_witness.py'
    extra=['--diagnostic-spec',str(ROOT/'configs/tracker-ppo-diagnostic.json'),
           '--references',str(RAM/'tracker_references/export_author_v1')]
cmd +=[str(ROOT/'scripts'/entry),'--spec',str(ROOT/'configs/tracker-env-spec.json'),
       '--output',str(out),'--device','cuda:1','--headless']+extra
log=ROOT/'logs'/('tracker_job_'+a.run+'.log');start=time.monotonic();timed_out=False
with log.open('xb') as target:
    child=subprocess.Popen(cmd,stdout=target,stderr=subprocess.STDOUT,start_new_session=True,
        preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
    try:code=child.wait(timeout=28800 if a.job in ['export643','export1653'] else 600)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof={'run':a.run,'job':a.job,'command':cmd,'child_pid':child.pid,'child_exit_code':code,
       'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,'numerical_status':receipt.get('status'),
       'full_training_started':False,'dataset_accepted':False,'scope':'finite reference export or bounded PPO diagnostic'}
(ROOT/'receipts'/('tracker_job_'+a.run+'_exit.json')).write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if code==0 and not timed_out and receipt.get('status')==expected:print('TEXTOP_TRACKER_JOB_EXIT0',flush=True)
else:print(log.read_text(errors='replace')[-7000:]);sys.exit(1)
