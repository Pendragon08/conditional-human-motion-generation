"""Wait for the already-running CPU conversion and audit its new outputs once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

W=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/reactive29_pairing_work_20261104')
D=W/'publisher60_20hz_full_v1'
receipt=W/'automatic_audit_launch.json'
receipt.open('x').write(json.dumps({'pid':os.getpid(),'watched_dataset':str(D),'purpose':'independent new-output audit, not training'}))
began=time.monotonic()
while not (D/'report.json').exists():
    status=json.loads((D/'status.json').read_text()) if (D/'status.json').exists() else {}
    if status.get('status')=='failed_preserved_prefix':
        (W/'automatic_audit_exit.json').open('x').write(json.dumps({'status':'build_failed_audit_not_started','build_status':status}));sys.exit(1)
    if time.monotonic()-began>1800:raise RuntimeError('bounded build wait expired')
    time.sleep(5)
report=json.loads((D/'report.json').read_text())
if report['status']!='paired_profile_built':
    (W/'automatic_audit_exit.json').open('x').write(json.dumps({'status':'partial_build_audit_not_started','failed_ids':report['failed_ids']}));sys.exit(1)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['nice','-n','10','timeout','--kill-after=20s','1800','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
    '-u',str(W/'audit_reactive29_paired_cpu_20261104.py'),'--dataset',str(D),'--output',str(W/'independent_audit_v1.json')]
with (W/'independent_audit_v1.log').open('x') as log:
    child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (W/'independent_audit_child.json').open('x').write(json.dumps({'pid':child.pid,'command':cmd}))
    code=child.wait()
(W/'automatic_audit_exit.json').open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
if code:sys.exit(code if code>0 else 1)
print('REACTIVE29_CPU_PAIRED_DATA_AUDIT_COMPLETED',flush=True)
