import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';source=B/'src/stage3_supported_postopt_guard_v3.py'
assert not (B/'pilot_postopt_v3_launch.json').exists() and not (B/'pilot_postopt_v3_guard.json').exists()
assert json.loads((B/'pilot_v2/audit_follow_status.json').read_text())['status']=='three_task_pilot_candidate_audit_passed_postoptimization_pending'
for p in [source,B/'src/stage3_supported_task_postopt_v3.py']:ast.parse(p.read_text())
log=(B/'pilot_postopt_v3_supervisor.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generated_candidates':5,'admitted_references':0,'purpose':'actual29DoF rotating/two-wrist optimization pilot; no formal training'}
(B/'pilot_postopt_v3_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
