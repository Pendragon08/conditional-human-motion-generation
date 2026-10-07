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
mon=r/'artifacts/monitoring'; art=r/'artifacts/world_confirmation_20261007'
name='r3_world_confirmation_20261007'
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
assert not (mon/(name+'.state')).exists()
assert not (art/'model_lock.json').exists()
assert not list(art.glob('collect_seed*_v1')) and not list(art.glob('evaluate_seed*_v1'))
assert not (mon/'r3_world_confirmation_collect_42_v1.state').exists()
oldlog=mon/(name+'.log'); assert oldlog.exists() and oldlog.stat().st_size==0
archive=mon/(name+'_crlf_dispatch_receipt.json'); assert not archive.exists()
archive.write_text(json.dumps({'status':'TRANSPORT_FAILED_BEFORE_QUEUE','old_log_bytes':0,
 'old_log_mtime':oldlog.stat().st_mtime,'no_queue_state_model_lock_or_data':True,
 'scientific_invocation_count':0,'actual_transitions':0,'cause':'PowerShell pipeline CRLF entered tmux redirection',
 'independent_witness':'world_confirmation_isaac_doc_witness_20261007','preserved_unix':time.time()},indent=2)+'\n')
script=r/'scripts/r3_world_confirmation_queue_20261007.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='c968485d24b97b014d9720c8d57a307751127a231995d14de8b310e7723a8b69'
cmd='cd '+str(r)+'; /usr/bin/python3 -u scripts/r3_world_confirmation_queue_20261007.py > artifacts/monitoring/r3_world_confirmation_20261007_transport_v2.log 2>&1'
assert '\r' not in cmd and '\n' not in cmd
subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
print('LF_SAFE_TM UX_DISPATCH_RETURNED'.replace('TM UX','TMUX'),flush=True)
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
