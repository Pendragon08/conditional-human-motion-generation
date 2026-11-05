import ast,json,os,subprocess,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'pilot_v2';source=B/'src/stage3_supported_pilot_audit_follow.py'
assert not (W/'audit_follow_launch.json').exists() and not (W/'audit_follow_status.json').exists()
ast.parse(source.read_text());log=(W/'audit_follow.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source':str(source)}
(W/'audit_follow_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
