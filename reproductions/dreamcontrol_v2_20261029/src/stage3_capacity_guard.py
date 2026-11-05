"""One bounded8912-environment capacity witness; monitor only this child process group."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
spec=json.loads((S/'env_spec_v2.json').read_text())
assert json.loads((S/'reports/physx_capacity_1024/status.json').read_text())['status']=='smoke_completed'
command=spec['smoke_command'];command[command.index('--num-envs')+1]='8912'
command[-1]=str(S/'reports/physx_capacity_8912')
assert not Path(command[-1]).exists()
receipt={'status':'running','command':command,'gpu_sum_limit_mib':12288,'host_rss_limit_gib':32,'max_seconds':300}
with (S/'reports/physx_capacity_8912.log').open('x') as log:
    process=subprocess.Popen(['timeout','300']+command,env={**os.environ,**spec['env']},cwd=spec['cwd'],
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['pid']=process.pid
    maximum_gpu=0;maximum_rss=0
    while process.poll() is None:
        own=[]
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():continue
            try:
                if os.getpgid(int(directory.name))==process.pid:own.append(int(directory.name))
            except ProcessLookupError:pass
        rss=0
        for pid in own:
            try:
                for line in Path('/proc/%d/status'%pid).read_text().splitlines():
                    if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
            except FileNotFoundError:pass
        output=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=10).stdout
        gpu=0
        for line in output.splitlines():
            fields=line.split(',')
            if len(fields)==2 and fields[0].strip().isdigit() and int(fields[0]) in own:
                try:gpu+=int(fields[1])
                except ValueError:pass
        maximum_gpu=max(maximum_gpu,gpu);maximum_rss=max(maximum_rss,rss)
        if gpu>12288 or rss>32*1024**3:
            receipt['status']='own_capacity_resource_limit'
            os.killpg(process.pid,signal.SIGTERM)
            break
        time.sleep(2)
    code=process.wait()
receipt.update(returncode=code,peak_own_gpu_mib=maximum_gpu,peak_own_host_rss_bytes=maximum_rss)
if receipt['status']=='running':receipt['status']='completed' if code==0 else 'failed_preserved_output'
with (S/'reports/physx_capacity_8912_guard.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
