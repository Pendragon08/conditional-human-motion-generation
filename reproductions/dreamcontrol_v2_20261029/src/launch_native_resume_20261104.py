import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';source=D/'src/native_tracker500_resume_guard.py';label='resume_8912_v1'
assert not (N/(label+'_launch.json')).exists() and not (N/(label+'_guard.json')).exists()
assert json.loads((N/'benchmark_8912_v1_guard.json').read_text())['status']=='passed'
assert json.loads((N/'evaluation_entry500_v1_guard.json').read_text())['status']=='passed'
for name in ['native_tracker500_resume_guard.py','native_tracker500_resume_worker.py']:ast.parse((D/'src'/name).read_text())
log=(N/(label+'_supervisor.log')).open('x')
p=subprocess.Popen(['python3','-u',str(source),'--mode','resume','--label',label],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'native actual optimizer resume; regenerated replay warmup explicitly counted'}
(N/(label+'_launch.json')).open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
