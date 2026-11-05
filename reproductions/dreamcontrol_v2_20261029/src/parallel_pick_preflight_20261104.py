import json,os,signal,hashlib,time,subprocess,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';D=R/'stage3_supermimic29_20261103'
W=D/'parallel_pick500_20261104';W.mkdir(exist_ok=True)
smoke=S/'reports/physx_capacity_8912/status.json'
s=json.loads(smoke.read_text());pid=s['pid']
cmd=Path('/proc')/str(pid)/'cmdline'
report={'previous_smoke':s,'free_bytes_before':shutil.disk_usage('/data').free}
if cmd.exists():
    raw=cmd.read_bytes().replace(b'\0',b' ').decode()
    assert s['status']=='smoke_completed' and s['steps']==32 and s['num_envs']==8912
    assert str(S/'src/stage3_pick29_run.py') in raw and str(smoke.parent) in raw and '--mode smoke' in raw
    report['completed_smoke_process_released']={'pid':pid,'command':raw,'status_sha256':hashlib.sha256(smoke.read_bytes()).hexdigest()}
    os.kill(pid,signal.SIGTERM)
    for _ in range(20):
        if not cmd.exists():break
        time.sleep(.5)
    if cmd.exists():os.kill(pid,signal.SIGKILL)
    report['completed_smoke_process_released']['still_exists_after']=cmd.exists()
report['ppo_status']=json.loads((S/'runs/pick500_8912_3000_v1/status.json').read_text())
assert report['ppo_status']['status']=='ppo_running'
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],universal_newlines=True)
report['ppo_checkpoint_sizes']={p.name:p.stat().st_size for p in (S/'runs/pick500_8912_3000_v1/ppo').glob('*.pt')}
report['free_bytes_after']=shutil.disk_usage('/data').free
report['project_sizes']=subprocess.check_output(['du','-x','-B1','--max-depth=2',str(R)],universal_newlines=True).splitlines()
(W/'preflight.json').open('x').write(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['previous_smoke','project_sizes']},indent=2))
print('\n'.join(report['project_sizes']))
