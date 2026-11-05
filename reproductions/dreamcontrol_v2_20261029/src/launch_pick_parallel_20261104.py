import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103'
W=D/'parallel_pick500_20261104'
sources=[D/'src/stage3_pick_parallel_controller.py',D/'src/stage3_supermimic_pick500_train.py']
for p in sources:ast.parse(p.read_text())
assert not (W/'controller_v1').exists() and not (W/'controller_v1_launch.json').exists()
assert json.loads((S/'runs/pick500_8912_3000_v1/status.json').read_text())['status']=='ppo_running'
log=(W/'controller_v1.log').open('x')
child=subprocess.Popen(['python3','-u',str(sources[0])],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':child.pid,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'policy':'Latest user requests parallel PPO+SuperMimic on separate GPUs; no PPO interruption or source modification.',
    'per_job_gpu_GiB':12,'per_job_RSS_GiB':32,'aggregate_gpu_GiB':24,'aggregate_RSS_GiB':64,
    'disk_floors_unchanged':True,'status':str(W/'controller_v1/status.json')}
(W/'controller_v1_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
