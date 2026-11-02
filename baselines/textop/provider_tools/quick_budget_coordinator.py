"""One bounded5k quick transition: wait, verify backups, stop this run, retain proof."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from session import remote

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.run)
import msvcrt
guard = (D/'quick_budget_coordinator.lock').open('a+b')
if guard.seek(0,2)==0:
    guard.write(b'0');guard.flush()
guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
folder = B/'results/quick_reproduction_20261026'/a.run
assert not folder.exists(), 'Use a new coordinator run after inspecting a prior failure'
folder.mkdir()
state = dict(status='STARTING',pid=os.getpid(),quick_target_iterations=5000,
             quality_label='QUICK_REPRODUCTION_NOT_PAPER_COMPLETE',data_deleted=False)
plan_path = B/'configs/quick-reproduction-plan-20261026.json'


def persist():
    state['sample_unix']=time.time()
    temp=folder/'state.partial';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(folder/'state.json')
    (D/'quick_reproduction_transition_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')


def control(action):
    expected=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    script="""import pathlib,hashlib,json,subprocess,sys
root=pathlib.Path(%r)
assert hashlib.sha256((root/'configs/quick-reproduction-plan-20261026.json').read_bytes()).hexdigest()==%r
r=subprocess.run([sys.executable,str(root/'scripts/quick_budget_control.py'),'--action',%r],capture_output=True,text=True,timeout=65)
assert r.returncode==0,(r.stderr,r.stdout)
print(r.stdout)
""" % (ROOT,expected,action)
    return json.loads(remote(script,timeout=85))


persist()
try:
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['early_stop_enabled'] is not True:
            state['status']='QUICK_TRANSITION_CANCELLED_BEFORE_STOP';persist();break
        observed=control('observe');state['observation']=observed
        if observed['status']=='USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE':
            state.update(status='QUICK_CHECKPOINT_READY_FOR_PILOT',controlled_stop=observed);persist();break
        if observed['status']=='QUICK_STOP_PENDING':
            state['status']='WAITING_ORIGINAL_SUPERVISOR_EXIT';persist()
            if observed['original_exit_observed']:
                stopped=control('finish')
                state.update(status='QUICK_CHECKPOINT_READY_FOR_PILOT',controlled_stop=stopped);persist();break
            time.sleep(10);continue
        if not observed['target_checkpoint_ready']:
            state['status']='WAITING_VERIFIED_5000_CHECKPOINT';persist();time.sleep(20);continue
        state['status']='BACKING_UP_FULL_PRIVATE_AND_PUBLIC_INFERENCE';persist()
        backup=subprocess.run([sys.executable,str(B/'provider_tools/backup_formal_filter_checkpoint.py'),'--completed','5000'],
                              capture_output=True,text=True,encoding='utf-8',timeout=1200)
        (folder/'backup_execution.json').write_text(json.dumps({'exit_code':backup.returncode,'stdout':backup.stdout,'stderr':backup.stderr},indent=2),encoding='utf-8')
        assert backup.returncode==0,backup.stderr
        verified_path=B/'results/formal_filter_checkpoints_20261026/round_5000/backup.json'
        verified=json.loads(verified_path.read_text(encoding='utf-8'))
        assert verified['status']=='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED'
        assert verified['audit']['full_checkpoint']['sha256']==observed['checkpoint']['sha256']
        script="""import pathlib,json,hashlib
p=pathlib.Path(%r)/'receipts/quick5000_backup_verified.json';data=%r
if p.exists():assert p.read_text()==data
else:
 with p.open('x') as f:f.write(data)
print(json.dumps({'backup_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
""" % (ROOT,verified_path.read_text(encoding='utf-8'))
        state['server_backup_proof']=json.loads(remote(script,timeout=30))
        if json.loads(plan_path.read_text(encoding='utf-8'))['early_stop_enabled'] is not True:
            state['status']='QUICK_TRANSITION_CANCELLED_AFTER_BACKUP_NO_STOP';persist();break
        state['status']='REQUESTING_USER_AUTHORIZED_CHECKPOINTED_EARLY_EXIT';persist()
        stopped=control('finish')
        state.update(status='QUICK_CHECKPOINT_READY_FOR_PILOT',controlled_stop=stopped,
                     next_step='Separate quick-policy real pilot validation; do not bypass40k formal gates')
        persist()
        for name in ['collect_formal_filter_progress.py','collect_quick_reproduction_report.py','collect_contact_report.py','publish.py']:
            recorded=subprocess.run([sys.executable,str(B/'provider_tools'/name)],capture_output=True,text=True,
                                    encoding='utf-8',timeout=300)
            (folder/(name+'.execution.json')).write_text(json.dumps({'exit_code':recorded.returncode,'stdout':recorded.stdout,'stderr':recorded.stderr},indent=2),encoding='utf-8')
            assert recorded.returncode==0,(name,recorded.stderr)
        break
    else:
        raise TimeoutError('Target checkpoint did not become ready in24h; trainer left running')
except BaseException:
    state['status']='QUICK_TRANSITION_NEEDS_INSPECTION'
    state['error']=traceback.format_exc()
    state['no_new_sdk_dispatched']=True
    persist()
    raise
