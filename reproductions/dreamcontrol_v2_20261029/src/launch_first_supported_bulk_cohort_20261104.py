import ast,hashlib,json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';source=B/'src/stage3_supported_candidate_cohort_guard.py'
assert not W.exists();W.mkdir();proofs=[B/'native_pilot_v1/report.json',B/'native_push_pilot_v1/report.json']
assert sum(json.loads(p.read_text())['admitted_references'] for p in proofs)==6
ast.parse(source.read_text());log=(W/'first_producer_supervisor.log').open('x');cohort=W/'cohorts/pour_0001'
command=['python3','-u',str(source),'--task','pour','--count','32','--start','11000000','--cohort',str(cohort)]
p=subprocess.Popen(command,cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'command':command,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cohort':str(cohort),'task':'pour','new_candidates':32,'new_candidate_ids':[11000000,11000031],'native_pilots_admitted':6,'accepted_library_target_per_task':500,'formal_training_started':False}
(W/'first_producer_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
