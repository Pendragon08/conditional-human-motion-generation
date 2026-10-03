"""Resume the original stopped scheduler after an audited zero-update failure."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
from r3_dreamcontrol_remaining_queue import ROOT, MON, TRAIN, PREFIX, digest

backup=ROOT/'artifacts/recovery/Open-Drawer_asset_20261003'
w=json.loads((MON/'r3_cabinet_asset_witness_20261003.json').read_text())
assert w['status']=='DIAGNOSTIC_PASS' and w['iterations']==1 and w['exit_code']==0
assert digest(TRAIN/'assets/cabinet.usd')==w['cabinet_sha256']
assert digest(Path(w['model']))==w['model_sha256'] and digest(Path(w['log']))==w['log_sha256']
ps=subprocess.check_output(['ps','-eo','args'],text=True)
assert not any('r3_i1_train_matched.py' in x or ('r3_dreamcontrol_remaining_queue.py' in x and '--check' not in x)
               for x in ps.splitlines() if not x.startswith(('sh ','bash ')))
session='r3_dreamcontrol_remaining_20261001'
assert subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0, 'Existing session requires inspection; no replacement'
log=MON/(PREFIX+'_Open-Drawer_train.log')
text=log.read_text(errors='replace')
assert 'Failed to find an articulation' in text
assert not re.search(r'Learning iteration\s+\d+/',text)
failed=TRAIN/'logs/rsl_rl/g1/2026-10-02_16-33-03-Open-Drawer-a'
assert not list(failed.glob('model_*.pt'))
assert digest(log)==digest(backup/'failed_train.log')
completed=['Jump','Sit','Pick','Bimanual-Pick','Ground-Pick','Ground-Pick-Top','Button-Press']
assert all((MON/(PREFIX+'_'+t+'.state')).read_text().startswith('COMPLETE') for t in completed)
archived=backup/'original_monitoring_train.log'
assert not archived.exists()
receipt=dict(status='FAILED_INITIALIZATION_RECOVERED', failed_run=str(failed),
             failed_log_sha256=digest(log), failed_formal_ppo_iterations=0,
             failed_formal_models=0, cabinet_sha256=w['cabinet_sha256'],
             independent_witness=w, skipped_completed_tasks=completed,
             remaining=['Open-Drawer','Open-Door'], formal_envs=8192,
             formal_iterations=2000, formal_seed=42,
             restarted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
log.rename(archived)
(backup/'resume_receipt.json').write_text(json.dumps(receipt,indent=2))
# Same original scheduler, same unique session and flock. No completed task retrained.
command='cd '+str(ROOT)+'; python3 -u scripts/r3_dreamcontrol_remaining_queue.py > artifacts/monitoring/r3_cabinet_recovered_queue_20261003.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
print('ORIGINAL_QUEUE_RESUMED',json.dumps(receipt),flush=True)
