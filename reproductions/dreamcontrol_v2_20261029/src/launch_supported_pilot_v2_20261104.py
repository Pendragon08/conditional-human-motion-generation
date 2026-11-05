import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'pilot_v2';source=B/'src/stage3_supported_pilot_queue_v2.py'
assert not W.exists() and not (B/'pilot_v2_launch.json').exists()
assert json.loads((B/'pilot_v1/status.json').read_text())['status']=='failed_preserved'
assert 'physical CUDA indices must be preserved' in (B/'pilot_v1/pour.log').read_text()
ast.parse(source.read_text());log=(B/'pilot_v2_controller.log').open('x')
p=subprocess.Popen(['python3','-u',str(source)],cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'count_per_task':2,'total_candidate_count':6,'accepted_references':0,'warm_reused_spec':'74af067a','fix':'preserve physical indices as required by unchanged existing setup; no visibility remap','previous_outputs_preserved':True}
(B/'pilot_v2_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
