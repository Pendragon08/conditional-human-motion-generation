
from pathlib import Path
import json,time,subprocess,datetime
r=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+7400
while time.monotonic()<deadline:
 p=r/'results/evaluation_suite.json'
 if p.is_file():
  state=json.loads(p.read_text())['status']
  if state=='FAILED':raise RuntimeError('Evaluation suite failed')
  if state=='COMPLETED':break
 time.sleep(20)
else:raise TimeoutError('Evaluation suite deadline')
for name in ['summarize_results.py','inspect_checkpoints.py']:
 subprocess.run(['bash','scripts/run_isaac.sh','scripts/'+name],cwd=r,check=True)
(r/'results/PIPELINE_DONE.json').write_text(json.dumps({'status':'COMPLETED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
