import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';source=B/'src/stage3_supported_bulk_optimizer_witness.py'
assert not (W/'optimization_witness').exists() and not (W/'optimization_witness_launch.json').exists()
for p in [source,B/'src/stage3_supported_bulk_postopt_v1.py']:ast.parse(p.read_text())
log=(W/'optimization_witness_supervisor.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reuse_prefix_candidate_ids':[11000000,11000001]};(W/'optimization_witness_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
