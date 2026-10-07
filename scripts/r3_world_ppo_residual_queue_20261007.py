"""One shared disturbance dataset, one bounded development fit per seed.

Stops after diagnosis; no launch of formal PPO or a reused confirmation test.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MON=ROOT/'artifacts/monitoring'
STATE=MON/'r3_world_residual_development_20261007.state'
ART=ROOT/'artifacts/world_residual_20261007'
SCRIPT=ROOT/'scripts/r3_world_ppo_residual_stage_20261007.sh'

def save(s):
    s['updated_unix']=time.time();s['formal_PPO_started']=False
    s['old_v1_FAIL_preserved']=True;s['retired_test_evaluations']=0
    f=STATE.with_suffix('.tmp');f.write_text(json.dumps(s,indent=2)+'\n');f.replace(STATE)

def main():
    with (MON/'r3_world_residual_development_20261007.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists(),'Unique queue; preserve existing state'
        probe=json.loads((ART/'probe_seed42_docwitness_v1/summary.json').read_text())
        assert probe['status']=='COMPLETE' and probe['actual_transitions']==7680
        assert probe['diagnostic_ppo_updates']==0 and probe['disturbance_protocol']['name']=='lateral80'
        kernel_env=dict(__import__('os').environ)
        kernel_env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        kernel_env['PYTHONPATH']=str(ROOT/'scripts')+':'+str(ROOT/'envs/dreamcontrol-r3/site-packages')
        save({'status':'RUNNING','stage':'seeded_residual_kernel_witness'})
        rc=subprocess.run([str(ROOT/'third_party/isaacsim-4.5.0/python.sh'),str(ROOT/'scripts/r3_world_ppo_residual_kernel_witness.py'),
                           '--output',str(ART/'kernel_witness.json')],env=kernel_env,timeout=300).returncode
        if rc:raise RuntimeError('Seeded residual witness failed: preserve and stop')
        complete=[];data=[]
        for seed in [42,43,44]:
            for mode in ['collect','fit']:
                if shutil.disk_usage(ROOT).free<15*1024**3:
                    save({'status':'BLOCKED','stage':mode,'seed':seed,'reason':'free_disk_under15GiB',
                          'completed':complete,'shared_datasets':data});return 1
                save({'status':'RUNNING','stage':mode,'seed':seed,'completed':complete,'shared_datasets':data})
                rc=subprocess.run(['bash',str(SCRIPT),mode,str(seed),'v1']).returncode
                if rc:
                    save({'status':'FAILED','stage':mode,'seed':seed,'exit':rc,'preserve':True,
                          'completed':complete,'shared_datasets':data});return 1
                f=ART/f'{mode}_seed{seed}_v1/summary.json';r=json.loads(f.read_text())
                digest=hashlib.sha256(f.read_bytes()).hexdigest()
                if mode=='collect':
                    assert r['actual_transitions']==3932160 and r['diagnostic_ppo_updates']==0
                    data.append({'seed':seed,'summary_sha256':digest,'transitions':r['actual_transitions'],
                                 'resets':r['reset_count'],'disturbance':r['disturbance_protocol']})
                else:
                    assert r['development_only'] and r['retired_test_evaluations']==0
                    assert r['aux_updates_per_model']==1000 and r['learned_model_count']==4
                    complete.append({'seed':seed,'summary_sha256':digest,
                                     'development_triage':r['development_triage'],
                                     'development_comparisons':r['development_comparisons'],'wall_s':r['wall_s']})
        promising=all(x['development_triage']=='PROMISING_NEEDS_NEW_CONFIRMATION_REFS' for x in complete)
        save({'status':'COMPLETE','stage':'development_only','completed':complete,'shared_datasets':data,
              'unique_new_transitions':sum(v['transitions'] for v in data),
              'innovation_validated':False,'formal_prediction_gate_passed':False,
              'next':'new_confirmation_references_and_local_view_gate' if promising else 'stop_residual_candidate',
              'development_triage':'PROMISING' if promising else 'FAIL'})
        print('WORLD_RESIDUAL_DEVELOPMENT_COMPLETE','PROMISING' if promising else 'FAIL',flush=True)
        return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        save({'status':'FAILED','stage':'queue','error':repr(e),'preserve':True})
        raise
