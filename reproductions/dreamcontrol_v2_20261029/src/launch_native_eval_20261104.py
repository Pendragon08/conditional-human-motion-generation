import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';source=D/'src/native_tracker500_eval_guard.py'
assert not (N/'evaluation_entry500_v1_launch.json').exists() and not (N/'evaluation_entry500_v1_guard.json').exists()
assert json.loads((N/'benchmark_8912_v1_guard.json').read_text())['status']=='passed'
for name in ['native_tracker500_eval_guard.py','native_tracker500_eval_worker.py']:ast.parse((D/'src'/name).read_text())
log=(N/'evaluation_entry500_v1_supervisor.log').open('x')
p=subprocess.Popen(['python3','-u',str(source)],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'500-body-reference evaluation entry, no learning'}
(N/'evaluation_entry500_v1_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
