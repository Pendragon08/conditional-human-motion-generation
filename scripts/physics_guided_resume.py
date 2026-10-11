"""Pause only this queue, let its separate child finish, then reconcile cost."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--pause-pid',type=int)
a=p.parse_args(); out=Path(a.output).resolve(); journal=out/'queue_pause_journal.json'
if a.pause_pid:
    command=Path('/proc/%d/cmdline'%a.pause_pid).read_bytes().replace(b'\0',b' ').decode()
    assert 'scripts/physics_guided_queue.py --output '+str(out) in command
    state=json.loads((out/'queue_state.json').read_text())
    journal.write_text(json.dumps(dict(state=state,pid=a.pause_pid,paused=time.time(),
        reason='Pause queue for implementation review; separate child may finish; preserve all completed artifacts'),indent=2))
    os.kill(a.pause_pid,signal.SIGTERM)
    print('QUEUE_ONLY_PAUSED_CHILD_NOT_KILLED')
else:
    data=json.loads(journal.read_text()); state=data['state']; label=state['label']
    receipt=out/'raw'/label/'receipt.json'; assert receipt.exists(), 'Wait for current child to finish'
    ledger_path=out/'budget_ledger.json'; ledger=json.loads(ledger_path.read_text())
    if not any(r.get('label')==label for r in ledger):
        ledger.append(dict(stage=state['stage'],label=label,seconds=time.time()-state['time'],
                           exit_code=0,status='conservative_wall_debit_after_queue_pause'))
        ledger_path.write_text(json.dumps(ledger,indent=2))
    data['reconciled']=time.time(); journal.write_text(json.dumps(data,indent=2))
    print('QUEUE_COST_RECONCILED',label)
