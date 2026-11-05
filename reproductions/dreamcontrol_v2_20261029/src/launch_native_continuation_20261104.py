import ast,hashlib,json,os,pathlib,subprocess
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104'
receipt=N/'continuation_launch.json'
assert not receipt.exists() and not (N/'continuation_status.json').exists()
assert json.loads((N/'smoke_author_v1_guard.json').read_text())['status']=='passed'
source=D/'src/native_tracker500_continuation.py'
for f in ['native_tracker500_continuation.py','native_tracker500_capacity_guard.py','native_tracker500_capacity_worker.py']: ast.parse((D/'src'/f).read_text())
log=(N/'continuation.log').open('x')
p=subprocess.Popen(['python3','-u',str(source)],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':p.pid,'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'serial8912capacity+compiledbenchmark; not formal training'}
receipt.open('x').write(json.dumps(data,indent=2));print(json.dumps(data))
