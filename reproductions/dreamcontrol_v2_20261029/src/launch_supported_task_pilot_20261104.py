import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'pilot_v1';source=B/'src/stage3_supported_pilot_queue.py'
assert not (W/'launch.json').exists() and not (W/'status.json').exists()
for p in (B/'src').glob('*.py'):ast.parse(p.read_text())
assert json.loads((W/'condition_preparation_report.json').read_text())['status']=='task_condition_plans_checked_not_generated'
log=(W/'controller.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'count_per_task':2,'total_candidate_count':6,'accepted_references':0,'scope':'first actual three-task pilot generation, not1500 batch or training'}
(W/'launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
