"""Collect each shared seed warm-up once. Never starts formal PPO."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MON=ROOT/'artifacts/monitoring'
STATE=MON/'r3_world_ppo_warmup_queue.state'
SCRIPT=ROOT/'scripts/r3_world_ppo_stage_20261006.sh'

def state(value):
    value['updated_unix']=time.time()
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)

def main():
    with (MON/'r3_world_ppo_warmup_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        witness=ROOT/'artifacts/world_ppo_20261006/smoke_seed42_docwitness_v2/summary.json'
        s=json.loads(witness.read_text())
        assert s['status']=='COMPLETE' and s['policy_weights_really_updated'] and s['world_weights_really_updated']
        assert not STATE.exists(), 'Unique queue: existing state must be audited, no duplicate'
        completed=[]
        for seed in [42,43,44]:
            free=shutil.disk_usage(ROOT).free
            if free<15*1024**3:
                state({'status':'BLOCKED','stage':'warmup','reason':'disk_free_under_15GiB',
                       'seed':seed,'completed':completed,'formal_PPO_started':False});return
            state({'status':'RUNNING','stage':'shared_frozen_policy_warmup','seed':seed,
                   'completed':completed,'formal_PPO_started':False,
                   'expected_transitions_per_seed':3932160,'free_disk_bytes':free})
            result=subprocess.run(['bash',str(SCRIPT),'collect',str(seed),'v1'])
            if result.returncode:
                state({'status':'FAILED','stage':'warmup','seed':seed,'exit':result.returncode,
                       'completed':completed,'preserve_outputs':True,'formal_PPO_started':False});return
            f=ROOT/f'artifacts/world_ppo_20261006/collect_seed{seed}_v1/summary.json'
            summary=json.loads(f.read_text())
            assert summary['actual_transitions']==3932160 and summary['diagnostic_ppo_updates']==0
            completed.append({'seed':seed,'transitions':summary['actual_transitions'],
                              'resets':summary['reset_count'],'wall_s':summary['wall_s'],
                              'summary_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        state({'status':'COMPLETE','stage':'shared_warmup_only','completed':completed,
               'actual_unique_transitions':11796480,'formal_PPO_started':False,
               'next':'documented_prediction_gate_then_local_visual_gate_then_matched_PPO'})

if __name__=='__main__':main()
