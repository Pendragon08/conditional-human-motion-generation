"""No formal RL launch; preserve first failure rather than endlessly retry."""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'artifacts/monitoring/r3_world_residual_ppo_smoke_20261007.state'
def write(data):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(STATE)

def main():
    assert not STATE.exists()
    completed=[]
    for arm in ['P0','P-cap','P-history','W-correct','W-shuffled-original']:
        write({'status':'RUNNING','stage':'diagnostic','arm':arm,'formal':False,'completed':completed,'updated_unix':time.time()})
        rc=subprocess.run(['bash',str(ROOT/'scripts/r3_world_residual_ppo_smoke_20261007.sh'),arm]).returncode
        if rc:
            write({'status':'FAILED','stage':'diagnostic','arm':arm,'wrapper_exit':rc,'formal':False,'completed':completed});return
        summary=json.loads((ROOT/f'artifacts/world_residual_ppo_20261007/smoke_{arm}_seed42_v1/summary.json').read_text())
        assert summary['status']=='COMPLETE' and summary['formal_PPO_updates']==0 and summary['diagnostic_PPO_updates']==2
        completed.append(arm)
    write({'status':'COMPLETE_DIAGNOSTICS','stage':'visibility','formal':False,'completed':completed,'updated_unix':time.time()})
    # Only frozen-policy real physics, bounded20minutes. User confirmation
    # may occur here; absence of a reply never starts formal PPO.
    rc=subprocess.run(['bash',str(ROOT/'scripts/r3_world_residual_visibility_20261007.sh')]).returncode
    write({'status':'COMPLETE' if rc==0 else 'FAILED_VISIBILITY','formal':False,'completed':completed,
           'visibility_wrapper_exit':rc,'formal_launch':'NOT_STARTED_REQUIRES_LOCAL_GATE_AND_RUNTIME_AUDIT'})
    print('WORLD_RESIDUAL_PPO_DIAGNOSTIC_QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
