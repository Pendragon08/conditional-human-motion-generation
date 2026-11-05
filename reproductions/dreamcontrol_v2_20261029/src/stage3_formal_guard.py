"""Monitor this single authorized PPO run; never restart it or signal other jobs."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
RUN=S/'runs/pick29_8912_5000_v1'
state={'pid':os.getpid(),'status':'monitoring','peak_own_gpu_mib':0,'peak_own_rss_bytes':0,
       'policy_run':str(RUN),'gpu_limit_mib':12288,'rss_limit_gib':32,'no_other_process_signals':True}


def save():
    p=S/'formal_guard_status.json';temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2));temp.replace(p)


def matching(pid):
    proc=Path('/proc/%d/cmdline'%pid)
    return proc.exists() and b'/stage3_pick29_20261103/src/stage3_pick29_run.py' in proc.read_bytes()


save()
began=time.monotonic()
while True:
    p=RUN/'status.json'
    if not p.exists():
        if time.monotonic()-began>300:
            state.update(status='launch_status_missing',log=str(S/'pick29_8912_5000_v1.log'));save();break
        time.sleep(5);continue
    try:run=json.loads(p.read_text())
    except json.JSONDecodeError:time.sleep(1);continue
    pid=run['pid'];state.update(run_pid=pid,policy_updates=run.get('policy_updates',0),run_status=run['status'])
    if run['status']=='failed':state.update(status='training_failed',failure=run.get('error'));save();break
    if run['status']=='ppo_completed':
        # A completed control run can hang in Isaac USD/renderer cleanup. Preserve
        # the completed status/checkpoints and release only this exact process.
        if matching(pid):
            time.sleep(30)
            if matching(pid):os.kill(pid,signal.SIGTERM);state['completed_cleanup_released']=True
        state.update(status='training_completed',policy_updates=run['policy_updates']);save();break
    if not matching(pid):state.update(status='process_missing_before_completion');save();break
    rss=0
    for line in Path('/proc/%d/status'%pid).read_text().splitlines():
        if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
    gpu=0
    try:
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        for line in query.stdout.splitlines():
            fields=line.split(',')
            if len(fields)==2 and fields[0].strip().isdigit() and int(fields[0])==pid:
                try:gpu+=int(fields[1])
                except ValueError:pass
    except subprocess.TimeoutExpired:pass
    state['peak_own_gpu_mib']=max(state['peak_own_gpu_mib'],gpu)
    state['peak_own_rss_bytes']=max(state['peak_own_rss_bytes'],rss)
    free=shutil.disk_usage('/data').free;state['free_disk_bytes']=free
    reason=('own_gpu_budget' if gpu>12288 else 'own_host_memory_budget' if rss>32*1024**3
            else 'disk_reserve' if free<8*1024**3+356*1024**2 else None)
    if reason:
        if matching(pid):os.kill(pid,signal.SIGINT)
        state.update(status='stopped_own_run_for_resource_protection',reason=reason);save();break
    save();time.sleep(15)
