import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
with (root/'reports/eyesjapan_stream_batch_status_20261103.jsonl').open('x',encoding='utf8',buffering=1) as f:
 f.write(json.dumps({'status':'running'})+'\n')
 rc=subprocess.call([sys.executable,str(root/'src/eyesjapan_stream_batch_producer_20261103.py'),'batch'])
 f.write(json.dumps({'status':'producer_exit','code':rc})+'\n')
 if rc:sys.exit(rc)
 cmd='cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029 && OMP_NUM_THREADS=1 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit_eyesjapan_stream_batch_20261103.py batch'
 rc=subprocess.call(['ssh','10.147.17.32',cmd])
 f.write(json.dumps({'status':'audit_exit_check_cases','code':rc})+'\n')
 if rc:sys.exit(rc)
