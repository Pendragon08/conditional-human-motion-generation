import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';source=B/'src/stage3_supported_completed_log_archive.py'
assert not (W/'log_archive_launch.json').exists() and not (W/'log_archive_status.json').exists();ast.parse(source.read_text())
log=(W/'log_archive_supervisor.log').open('x');p=subprocess.Popen(['nice','-n','10','python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'verified lossless compression of own completed closed logs only; no source arrays/model/checkpoint changes'}
(W/'log_archive_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
