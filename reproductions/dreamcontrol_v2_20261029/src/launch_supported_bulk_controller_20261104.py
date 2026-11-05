import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';source=B/'src/stage3_supported_bulk_controller.py'
assert not (W/'controller_launch.json').exists() and not (W/'controller_status.json').exists()
for name in ['stage3_supported_bulk_controller.py','stage3_supported_bulk_postopt_v1.py','stage3_supported_bulk_audit.py','stage3_supported_native_replay.py']:ast.parse((B/'src'/name).read_text())
assert json.loads((W/'optimization_witness/status.json').read_text())['status']=='production_optimizer_first_two_completed_reuse_required'
assert json.loads((W/'optimization_witness/postopt/report.json').read_text())['completed']==2
log=(W/'controller.log').open('x');p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'task_targets':{'pour':500,'bimanual_box_lift':500,'box_push_pull':500},'cohort_size':32,'first_producer_adopted':1479787,'reuse_optimizer_prefix':[11000000,11000001],'formal_training_started':False}
(W/'controller_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
