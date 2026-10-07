"""One read-only SSH metadata fetch, no experiment or licensed array transfer."""
import base64
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SSH=['ssh','-b','10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519',
     '-o','BatchMode=yes','-o','ConnectTimeout=15','ouyanghao@10.147.17.32']
CODE=r'''
from pathlib import Path
import json,base64,time
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
m=r/'artifacts/monitoring';a=r/'artifacts/world_confirmation_20261007'
v2=(m/'r3_world_confirmation_v2_20261007.state').exists()
prefix='r3_world_confirmation_v2' if v2 else 'r3_world_confirmation'
current=a/'confirmation_v2' if v2 else a
paths={'queue.state':m/(prefix+'_20261007.state'),
 'model_lock.json':current/'model_lock.json',
 'v1_queue.state':m/'r3_world_confirmation_20261007.state',
 'v1_model_lock.json':a/'model_lock.json',
 'v1_collect_seed42.state':m/'r3_world_confirmation_collect_42_v1.state',
 'failed_initializer_cleanup_termination.json':m/'r3_world_confirmation_v1_failed_cleanup_termination.json',
 'transport_failed_receipt.json':m/'r3_world_confirmation_20261007_crlf_dispatch_receipt.json'}
for seed in [42,43,44]:
 for mode in ['collect','evaluate']:
  paths[f'{mode}_seed{seed}_summary.json']=current/f'{mode}_seed{seed}_v1/summary.json'
  paths[f'{mode}_seed{seed}.state']=m/f'{prefix}_{mode}_{seed}_v1.state'
  paths[f'{mode}_seed{seed}_coverage.json']=current/f'{mode}_seed{seed}_v1/coverage.json'
logs=[m/'r3_world_confirmation_20261007_transport_v2.log',m/'r3_world_confirmation_v2_20261007.log']
logs+=[m/f'{prefix}_{mode}_{seed}_v1.log' for mode in ['collect','evaluate'] for seed in [42,43,44]]
out={name:base64.b64encode(p.read_bytes()).decode() for name,p in paths.items() if p.exists()}
print(json.dumps({'receipts':out,'observed_unix':time.time(),
 'log_tails':{p.name:{'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,
 'tail':p.read_text(errors='replace')[-2000:]} for p in logs if p.exists()}}))
'''


def main():
    encoded=base64.b64encode(CODE.encode('utf-8')).decode('ascii')
    cmd='/usr/bin/python3 -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    p=subprocess.run(SSH+[cmd],capture_output=True,text=True)
    if p.returncode:
        print(p.stderr)
        raise SystemExit(p.returncode)
    data=json.loads(p.stdout)
    raw=ROOT/'results/world_confirmation_20261007/raw'
    raw.mkdir(parents=True,exist_ok=True)
    for name, encoded in data['receipts'].items():
        (raw/name).write_bytes(base64.b64decode(encoded))
    # The observation is not a completion receipt; retain tails for diagnosis.
    (raw/'latest_read_only_observation.json').write_text(json.dumps({
        'observed_unix':data['observed_unix'],'log_tails':data['log_tails'],
        'downloaded_receipt_names':list(data['receipts'])},indent=2)+'\n',encoding='utf-8')
    state=json.loads((raw/'queue.state').read_text()) if (raw/'queue.state').exists() else {}
    print('CONFIRMATION_READONLY',state.get('status','NO_QUEUE_STATE'),state.get('stage'),state.get('seed'),
          len(state.get('completed',[])),list(data['receipts']))
    if state.get('status')=='FAILED':
        print(json.dumps(data['log_tails'],indent=2))


if __name__=='__main__':main()
