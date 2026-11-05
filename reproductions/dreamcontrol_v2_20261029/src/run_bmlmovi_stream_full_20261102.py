"""Run new full stream then same-model witness audit, preserving all logs."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
status=root/'reports/bmlmovi_stream_full_status_20261102.jsonl'
with status.open('x',encoding='utf8',buffering=1) as log:
 log.write(json.dumps({'status':'running'})+'\n')
 rc=subprocess.call([sys.executable,str(root/'src/bmlmovi_stream_producer_20261102.py'),'full'])
 log.write(json.dumps({'status':'producer_exit','code':rc})+'\n')
 if rc:sys.exit(rc)
 command='cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029 && OMP_NUM_THREADS=1 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit_bmlmovi_stream_20261102.py full'
 rc=subprocess.call(['ssh','10.147.17.32',command])
 log.write(json.dumps({'status':'audit_exit_check_cases','code':rc})+'\n')
 if rc:sys.exit(rc)
