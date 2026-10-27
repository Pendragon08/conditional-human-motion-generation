"""Bounded document-following milestone; refuse reruns/overwriting evidence."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/prefix_integrity/document_run'
PARENT = ROOT/'logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == 'bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336'
    record = {'status':'RUNNING','purpose':'bounded engineering validation; not full reproduction',
              'stages':[], 'started_unix':time.time()}
    path = OUT/'pipeline.json'
    def stage(name, command, timeout=600):
        record['stages'].append({'name':name,'command':command,'status':'RUNNING'})
        path.write_text(json.dumps(record,indent=2)+'\n')
        code = subprocess.call([sys.executable,'scripts/run_prefix_recorded.py','--name',
            'prefix_integrity/document_run/'+name,'--timeout',str(timeout),'--',*command],cwd=ROOT)
        record['stages'][-1].update(exit_code=code,status='COMPLETED' if code == 0 else 'FAILED')
        if code:
            record['status']='FAILED'
            path.write_text(json.dumps(record,indent=2)+'\n')
            raise SystemExit(code)
        path.write_text(json.dumps(record,indent=2)+'\n')
    stage('prepare',['python3','scripts/prepare_guarded_upstream.py','--destination','cache/guarded_upstream'],120)
    stage('regression',['bash','scripts/run_isaac.sh','reimplementation/test_prefix_and_curriculum.py',
        '--output','results/prefix_integrity/document_run/regression/verification.json'],120)
    stage('one_ppo',['bash','scripts/run_isaac.sh','scripts/guarded_bootstrap.py',
        '--task','Tracking-Flat-G1-Wo-State-Estimation-v0','--headless','--device','cuda:0',
        '--num_envs','32','--seed','42','--max_iterations','1','--run_name','prefix_guard_diagnostic',
        '--logger','tensorboard','--g1_training_cfg','configs/moe_pilot.yaml',
        '--motion_file','data/train/sample_000_actor0.npz','--no_wandb_model_upload'])
    stage('feedback_capture',['bash','scripts/run_isaac.sh','scripts/evaluate_prefix_feedback.py',
        '--checkpoint',str(PARENT),'--motion_file','data/train/sample_000_actor0.npz',
        '--output','results/prefix_integrity/document_run/capture','--num_envs','32',
        '--seed','2026','--headless','--device','cuda:0'])
    stage('feedback_gate',['bash','scripts/run_isaac.sh','scripts/diagnose_prefix_feedback.py',
        '--capture','results/prefix_integrity/document_run/capture',
        '--output','results/prefix_integrity/document_run/feedback_gate/verification.json'],120)
    record.update(status='COMPLETED',finished_unix=time.time())
    path.write_text(json.dumps(record,indent=2)+'\n')
    print('PHYGILE_PREFIX_MILESTONE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
