"""LF-independent SSH dispatch; reject unknown or existing scientific runs."""
from pathlib import Path
import base64
import json
import shlex
import subprocess

SSH = ['ssh','-b','10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519',
       '-o','BatchMode=yes','-o','ConnectTimeout=15','ouyanghao@10.147.17.32']
REMOTE = '/data/code/ouyanghao/oyh/conditional-human-motion-generation'
REMOTE_PROGRAM = r'''
from pathlib import Path
import hashlib,json,subprocess,time
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
mon=r/'artifacts/monitoring';base=r/'artifacts/world_confirmation_20261007';art=base/'confirmation_v2'
name='r3_world_confirmation_v2_20261007'
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
assert not (mon/(name+'.state')).exists() and not art.exists()
old=json.loads((mon/'r3_world_confirmation_20261007.state').read_text())
assert old['status']=='FAILED' and old['stage']=='collect' and old['seed']==42
assert not old['completed'] and not old['datasets']
assert not list((base/'collect_seed42_v1').glob('*')) and not (base/'evaluate_seed42_v1').exists()
script=r/'scripts/r3_world_confirmation_v2_queue_20261007.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='03791432f0765da982438658d0e608f5fba4da2f2bd582837c34b31470d7801b'
cmd='cd '+str(r)+'; /usr/bin/python3 -u scripts/r3_world_confirmation_v2_queue_20261007.py > artifacts/monitoring/r3_world_confirmation_v2_20261007.log 2>&1'
assert '\r' not in cmd and '\n' not in cmd
subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
print('WORLD_CONFIRMATION_V2_DISPATCH_RETURNED',flush=True)
'''


def main():
    payload = base64.b64encode(REMOTE_PROGRAM.encode('utf-8')).decode('ascii')
    python = 'import base64;exec(base64.b64decode('+repr(payload)+'))'
    command = '/usr/bin/python3 -c ' + shlex.quote(python)
    receipt = subprocess.run(SSH+[command],capture_output=True,text=True)
    print(receipt.stdout,end='')
    print(receipt.stderr,end='')
    # A nonzero/timeout or unknown outcome must be inspected read-only, not retried.
    raise SystemExit(receipt.returncode)


if __name__=='__main__':
    main()
