import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';source=B/'src/stage3_supported_native_push_guard_v1.py'
assert not (B/'native_push_pilot_v1_launch.json').exists() and not (B/'native_push_pilot_v1').exists()
for p in [source,B/'src/stage3_supported_native_replay.py']:ast.parse(p.read_text())
log=(B/'native_push_pilot_v1_supervisor.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'purpose':'native task-specific geometry admission of four preserved robot pilots; not training'}
(B/'native_push_pilot_v1_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
