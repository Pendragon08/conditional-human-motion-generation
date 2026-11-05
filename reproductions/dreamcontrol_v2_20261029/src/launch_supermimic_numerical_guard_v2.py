import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';W=D/'parallel_pick500_20261104';S=R/'stage3_pick29_20261103'
source=D/'src/stage3_supermimic_numerical_guard_v2.py';ast.parse(source.read_text())
assert json.loads((D/'runs/pick500_8912_72000_parallel_v1/status.json').read_text())['status']=='failed'
assert json.loads((S/'runs/pick500_8912_3000_v1/status.json').read_text())['status']=='ppo_running'
assert not (W/'numerical_repair_v2').exists() and not (W/'numerical_repair_v2_launch.json').exists()
log=(W/'numerical_repair_v2.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':str(W/'numerical_repair_v2/status.json'),'PPO_guard_unchanged':1442525}
(W/'numerical_repair_v2_launch.json').open('x').write(json.dumps(r,indent=2));print(json.dumps(r))
