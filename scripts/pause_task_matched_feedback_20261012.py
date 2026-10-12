"""Explicit user pause of this exact experiment process tree; no deletion."""
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone

root=Path(__file__).resolve().parents[1]
out=root/'results/task_matched_feedback_continue_20261012'
pid=2935169
expected='scripts/task_matched_feedback_continue_20261012.py --run'
record=dict(user_requested_pause=True,expected_parent_pid=pid,expected_command_fragment=expected,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),deleted_files=0,restarts=0)
cmd=Path('/proc/%d/cmdline'%pid)
if cmd.exists():
    actual=cmd.read_bytes().replace(b'\0',b' ').decode()
    if expected not in actual: raise RuntimeError('PID no longer belongs to exact experiment')
    record['actual_parent_command']=actual
    os.kill(pid,signal.SIGSTOP)
    table={}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            line=(folder/'stat').read_text(); rest=line[line.rindex(')')+2:].split()
            table[int(folder.name)]=int(rest[1])
        except (OSError,ValueError,IndexError): pass
    descendants={pid}
    while True:
        old=len(descendants)
        descendants.update(p for p,parent in table.items() if parent in descendants)
        if len(descendants)==old: break
    stopped=[]
    for child in sorted(descendants-{pid},reverse=True):
        try: os.kill(child,signal.SIGTERM); stopped.append(child)
        except ProcessLookupError: pass
    # Queue SIGINT before continuing: Python finally retains the interrupted
    # batch debit; no automatic next batch or destructive cleanup is invoked.
    os.kill(pid,signal.SIGINT)
    os.kill(pid,signal.SIGCONT)
    record['terminated_descendant_pids']=stopped
    for _ in range(100):
        if not cmd.exists(): break
        time.sleep(.1)
else:
    record['parent_missing_before_pause']=True
record['parent_exists_after_wait']=cmd.exists()
record['preserved_ledger']=json.loads((out/'budget_ledger.json').read_text())
destination=out/('USER_PAUSE_%d.json'%time.time_ns())
with destination.open('x',encoding='utf8') as f: json.dump(record,f,indent=2)
print(json.dumps(dict(receipt=str(destination),parent_exists=record['parent_exists_after_wait'],ledger=record['preserved_ledger'])),flush=True)
