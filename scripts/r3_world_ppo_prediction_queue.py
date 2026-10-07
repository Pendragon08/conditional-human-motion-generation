"""Bounded prediction gate; does not launch formal control training."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MON=ROOT/'artifacts/monitoring'
STATE=MON/'r3_world_ppo_prediction_queue.state'
ART=ROOT/'artifacts/world_ppo_20261006'

def save(data):
    data['updated_unix']=time.time()
    f=STATE.with_suffix('.tmp');f.write_text(json.dumps(data,indent=2)+'\n');f.replace(STATE)

def main():
    with (MON/'r3_world_ppo_prediction_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists(), 'Preserve existing prediction queue'
        save({'status':'WAITING','stage':'shared_warmup_completion','formal_PPO_started':False})
        deadline=time.time()+1800
        while time.time()<deadline:
            f=MON/'r3_world_ppo_warmup_queue.state'
            s=json.loads(f.read_text()) if f.exists() else {}
            if s.get('status')=='COMPLETE':break
            if s.get('status') in ['FAILED','BLOCKED']:raise RuntimeError('Shared data queue did not complete')
            time.sleep(10)
        else:raise RuntimeError('Bounded 1800s warmup wait expired')
        assert s['actual_unique_transitions']==11796480
        completed=[]
        for seed in [42,43,44]:
            save({'status':'RUNNING','stage':'prediction_fitting_and_locked_test',
                  'seed':seed,'completed':completed,'formal_PPO_started':False})
            rc=subprocess.run(['bash',str(ROOT/'scripts/r3_world_ppo_stage_20261006.sh'),
                               'fit',str(seed),'v1']).returncode
            if rc:
                save({'status':'FAILED','stage':'prediction','seed':seed,'exit':rc,
                      'completed':completed,'formal_PPO_started':False});return 1
            f=ART/f'fit_seed{seed}_v1/summary.json';r=json.loads(f.read_text())
            completed.append({'seed':seed,'prediction_gate':r['prediction_gate'],
                              'paired_H10_gate':r['paired_H10_gate'],
                              'summary_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                              'wall_s':r['wall_s']})
        gate='PASS' if all(x['prediction_gate']=='PASS' for x in completed) else 'FAIL'
        save({'status':'COMPLETE','prediction_gate':gate,'completed':completed,
              'formal_PPO_started':False,'innovation_validated':False,
              'next':'formal_code_and_current_view_gate' if gate=='PASS' else 'diagnose_prediction_failure_before_PPO'})
        print('WORLD_PPO_ALL_SEED_PREDICTION_COMPLETE',gate,flush=True)
        return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        save({'status':'FAILED','stage':'prediction_queue','error':repr(e),
              'formal_PPO_started':False,'preserve':True})
        raise
