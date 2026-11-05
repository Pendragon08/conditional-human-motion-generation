import json,os,subprocess,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';W=D/'native_universal_20261104'
assert not (W/'smoke_author_v1').exists() and not (W/'smoke_author_v1_launch.json').exists()
assert json.loads((W/'preparation_receipt.json').read_text())['references']==500
fresh=json.loads((W/'kernel_freshdoc_v1_guard.json').read_text());assert fresh['status']=='passed'
log=(W/'smoke_author_v1_guard.log').open('x');source=D/'src/native_tracker500_guard.py'
p=subprocess.Popen(['python3','-u',str(source),'--mode','smoke','--label','smoke_author_v1'],cwd=str(D),
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'guard_status':str(W/'smoke_author_v1_guard.json'),
 'mode':'full upstream native train.py, real500 library, compiled WM/BPTT and2048-sized batches; bounded5 iterations, not formal Pick training'}
(W/'smoke_author_v1_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
