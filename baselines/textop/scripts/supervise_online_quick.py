"""Owns one exclusive joint experiment: one generator plus one SDK, aggregate Torch cap .65."""
import argparse,fcntl,json,os,pathlib,signal,subprocess,time,traceback
from online_quick_protocol import ROOT,sha,write,load_spec,request,response
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
from quick_pidfd import BoundProcess,start_ticks
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();s,sh=load_spec();assert a.run.isidentifier()
os.environ.update(OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0')
lk=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text())
assert all(sha(ROOT/n)==h for n,h in dict(plan['frozen_runtime_files'],**plan['control_sources']).items())
sample_resources(ROOT,None,20*1024**3,3*1024**3)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};assert free[1]>20480
blocked=['run_tracker.sh','run_quick_ldm.py','run_quick_vae.py','run_generated_pipeline_sdk.py','online_quick_generator.py','online_quick_sdk.py']
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        if proc.stat().st_uid!=1013:continue
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not {str(ROOT/'scripts'/n) for n in blocked}.intersection(argv),'Existing active job, no overlapping SDK or generator'
out=ROOT/'online_private'/a.run;out.mkdir(mode=0o700,parents=True,exist_ok=False)
r={'status':'STARTING_JOINT_ONLINE','spec_sha256':sh,'run':a.run,'optimizer_updates':0,'sdk_process_count':0,'generator_process_count':0,'writers_closed':False,'quality_accepted':False}
gen=None;fd=None;glog=(out/'generator.log').open('xb');sdkproof={}
try:
    request(out,0,0,4)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'scripts'))
    cmd=[str(ROOT/'env/bin/python'),'-u',str(ROOT/'scripts/online_quick_generator.py'),'--run',a.run]
    gen=subprocess.Popen(cmd,env=env,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{0}));fd=BoundProcess(gen.pid,start_ticks(gen.pid))
    r.update(generator_pid=gen.pid,generator_process_count=1);write(out/'pipeline.json',r)
    until=time.monotonic()+90
    while response(out,0) is None:
        assert gen.poll() is None,'Generator exited before bootstrap';assert time.monotonic()<until,'Bootstrap timeout';time.sleep(.2)
    cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/online_quick_sdk.py'),'--run',a.run,'--device','cuda:1','--headless']
    def both(pid):
        sample=sample_resources(ROOT,pid,20*1024**3,3*1024**3)
        sample['joint_generator']=sample_resources(ROOT,gen.pid,20*1024**3,3*1024**3)
        assert gen.poll() is None,'Generator exited during joint experiment'
        return sample
    r['sdk_process_count']=1;write(out/'pipeline.json',r)
    sdkproof=run_supervised(cmd,out/'sdk.log',out/'resources.jsonl',out/'sdk_process_exit.json',both,s['sdk_timeout_seconds'],15,affinity=(1,2,3))
    write(out/'stop.json',{'reason':'SDK child reaped','at':time.monotonic()})
    code=gen.wait(timeout=30);glog.flush();os.fsync(glog.fileno())
    gr=json.loads((out/'generator_receipt.json').read_text());sr=json.loads((out/'sdk/receipt.json').read_text())
    r.update(generator_exit_code=code,sdk_process_proof=sdkproof,generator_receipt_sha256=sha(out/'generator_receipt.json'),sdk_receipt_sha256=sha(out/'sdk/receipt.json'))
    assert code==0 and gr['writers_closed'] and sdkproof['supervision_passed'] and sr['writers_closed'],r
    r.update(status='JOINT_ONLINE_EXPERIMENT_CLOSED',writers_closed=True,actual_control_calls=sr['actual_control_calls'],physical_failure=sr['physical_failure'],
        hot_command_consumed=sr['switched_target_consumed'] is not None,all_deadlines_passed=sr['actual_50hz_deadlines_passed'],deadline_misses=sr['deadline_misses'],
        full_bounded_duration=sr['full_bounded_duration'],stream_underruns=sr['stream_underruns'],quality_accepted=False,
        combined_peak_torch_upper_bound_bytes=gr['peak_torch_allocated_bytes']+sr['peak_torch_allocated_bytes'],
        native_close_returned=sr['native_close_returned'],teardown_fallback_used=sr['teardown_fallback_used'])
except BaseException as e:r.update(status='FAILED_JOINT_ONLINE',error=repr(e),traceback=traceback.format_exc());raise
finally:
    write(out/'stop.json',{'reason':'joint supervisor finally','at':time.monotonic()})
    if gen is not None and gen.poll() is None:
        try:gen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fd.send(signal.SIGTERM);r['generator_stop_signal']='SIGTERM'
            try:gen.wait(timeout=10)
            except subprocess.TimeoutExpired:fd.send(signal.SIGKILL);r['generator_stop_signal']='SIGKILL';gen.wait(timeout=10)
    if gen is not None:r['actual_generator_returncode']=gen.returncode
    if fd is not None:fd.close()
    glog.close();write(out/'pipeline.json',r);print(json.dumps(r),flush=True)
