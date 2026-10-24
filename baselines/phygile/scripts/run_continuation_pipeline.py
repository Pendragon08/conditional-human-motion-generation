"""Bounded continuation and fixed-checkpoint evaluation; no test-driven selection."""
from pathlib import Path
import datetime,json,subprocess,sys,traceback
root=Path(__file__).resolve().parents[1]
out=root/'results/continuation_2000';out.mkdir(exist_ok=True)
statefile=out/'pipeline.json';assert not statefile.exists(), 'Refusing to overwrite a prior pipeline'
state={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':[]}
statefile.write_text(json.dumps(state,indent=2)+'\n')
commands=[
    [sys.executable,'scripts/run_recorded.py','--name','continuation_2000/train','--timeout','10800','--','bash','scripts/train_extended_2000.sh'],
    [sys.executable,'scripts/evaluate_continuation.py'],
    ['bash','scripts/run_isaac.sh','scripts/summarize_continuation.py']
]
try:
    for command in commands:
        result=subprocess.run(command,cwd=root)
        state['stages'].append({'command':command,'exit_code':result.returncode})
        statefile.write_text(json.dumps(state,indent=2)+'\n')
        if result.returncode:raise RuntimeError('Pipeline stage failed')
    state['status']='COMPLETED'
except BaseException:
    state['status']='FAILED';state['error']=traceback.format_exc();raise
finally:
    state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    statefile.write_text(json.dumps(state,indent=2)+'\n')
