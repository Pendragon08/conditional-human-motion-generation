"""Real signals target only a newly created CPU fixture, never any training PID."""
import json
import pathlib
import signal
import subprocess
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from quick_pidfd import BoundProcess, start_ticks

child = subprocess.Popen([sys.executable,'-u','-c',
    'import signal,time,sys;signal.signal(signal.SIGINT,lambda s,f:sys.exit(17));print("READY",flush=True);time.sleep(30)'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
sentinel = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
checks=[]
try:
    assert child.stdout.readline().strip()=='READY'
    born=start_ticks(child.pid)
    try:
        BoundProcess(child.pid,born+1)
    except AssertionError:
        checks.append({'case':'changed_start_identity_refused','passed':True})
    else:raise AssertionError('Changed process identity accepted')
    handle=BoundProcess(child.pid,born)
    handle.send(signal.SIGINT)
    assert child.wait(timeout=5)==17 and not handle.alive()
    assert sentinel.poll() is None
    checks.append({'case':'bound_cpu_target_received_sigint','passed':True})
    try:handle.send(signal.SIGINT)
    except ProcessLookupError:
        checks.append({'case':'exited_original_cannot_signal_another_process','passed':True})
    else:raise AssertionError('Stale original process accepted another signal')
    handle.close()
finally:
    for proc in [child,sentinel]:
        if proc.poll() is None:proc.terminate()
        proc.wait(timeout=5)
print(json.dumps({'status':'QUICK_PIDFD_CPU_FIXTURE_PASSED','checks':checks,'signals_to_training':0,'sdk_runs':0,'gpu_jobs':0,'sentinel_unaffected':True}))
