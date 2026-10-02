"""Separate, bounded frozen-policy transfer queue; never edits baseline training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'third_party/DreamControl/Training'
MON=ROOT/'artifacts/monitoring'
PREFIX='r3_sim2sim_completed_20261002'
RUNS={'Jump':'2026-10-01_14-01-43-Jump-a','Sit':'2026-10-01_16-54-27-Sit-a',
      'Pick':'2026-10-01_19-59-36-Pick-a','Bimanual-Pick':'2026-10-02_00-21-24-Bimanual-Pick-a'}
PY='/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def environment():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        WANDB_MODE='disabled',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',
        TMPDIR=str(ROOT/'cache/r3_tmp'),XDG_CACHE_HOME=str(ROOT/'cache/r3_xdg'),
        MPLCONFIGDIR=str(ROOT/'cache/r3_matplotlib'))
    env['PYTHONPATH']=':'.join(str(p) for p in [ROOT/'scripts',ROOT/'envs/dreamcontrol-r3/site-packages',
        TRAIN/'source/isaaclab',TRAIN/'source/isaaclab_assets',TRAIN/'source/isaaclab_tasks',
        TRAIN/'source/isaaclab_rl',TRAIN/'isaac_utils',TRAIN/'scripts/reinforcement_learning/rsl_rl'])
    env.pop('CUDA_VISIBLE_DEVICES',None)
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['check','export','witness','eval','all'],default='check')
    p.add_argument('--task',choices=list(RUNS));p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    assert a.attempt in [1,2], 'At most one explicitly audited diagnostic correction'
    tasks=[a.task] if a.task else list(RUNS)
    spec=json.loads((ROOT/'configs/mujoco_sim2sim_env.json').read_text())
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
    assert digest=='91455383' and Path(PY).is_file()
    for task in tasks:
        run=TRAIN/'logs/rsl_rl/g1'/RUNS[task]
        record=json.loads((ROOT/f'results/r3_dreamcontrol_official_{task}_8192_eval.json').read_text())
        assert sha(run/'model_1999.pt')==record['checkpoint_sha256']
        assert (run/'exported/policy.pt').is_file() and record['train_exit']==record['eval_exit']==0
    print('COMPLETED_SIM2SIM_PREFLIGHT_PASS tasks='+','.join(tasks)+' env_hash='+digest,flush=True)
    if a.stage=='check':return
    MON.mkdir(parents=True,exist_ok=True)
    with (MON/(PREFIX+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=['export','witness','eval'] if a.stage=='all' else [a.stage]
        for task in tasks:
            for stage in stages:
                tag=f'{PREFIX}_{task}_{stage}'+(f'_attempt{a.attempt}' if a.attempt>1 else '')
                state=MON/(tag+'.state');log=MON/(tag+'.log')
                if state.exists() and state.read_text().startswith('COMPLETE'):continue
                if log.exists():raise RuntimeError('Existing attempt must be audited, not overwritten: '+str(log))
                (MON/(PREFIX+'.state')).write_text(f'RUNNING task={task} stage={stage}\n')
                state.write_text(f'RUNNING task={task} stage={stage} learning=false\n')
                if stage=='export':
                    cmd=[str(ROOT/'third_party/isaacsim-4.5.0/python.sh'),'-u',str(ROOT/'scripts/r3_sim2sim_completed_export.py'),
                        '--task',task,'--policy',str(TRAIN/'logs/rsl_rl/g1'/RUNS[task]/'exported/policy.pt'),
                        '--output',str(ROOT/'artifacts/sim2sim_mujoco/interfaces'),'--device','cuda:0','--headless','--disable_fabric']
                    sentinel='COMPLETED_TASK_ISAAC_EXPORT';cwd=TRAIN;seconds=600
                else:
                    cmd=[PY,'-u',str(ROOT/'scripts/r3_sim2sim_completed_mujoco.py'),'--task',task,'--mode',stage]
                    sentinel='COMPLETED_SIM2SIM_WITNESS' if stage=='witness' else 'COMPLETED_SIM2SIM_EVAL'
                    cwd=ROOT;seconds=3600
                with log.open('wb') as f:
                    proc=subprocess.Popen(cmd,cwd=cwd,env=environment(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    try:rc=proc.wait(timeout=seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        rc=124
                if rc or sentinel not in log.read_text(errors='replace'):
                    state.write_text(f'FAILED task={task} stage={stage} exit={rc}\n')
                    (MON/(PREFIX+'.state')).write_text(f'BLOCKED task={task} stage={stage} exit={rc} log={log}\n')
                    raise RuntimeError(f'{task} {stage} failed exit={rc}; log={log}')
                state.write_text(f'COMPLETE task={task} stage={stage} exit=0\n')
        if a.stage=='witness' and not a.task:
            log=MON/(PREFIX+'_reference_audit.log')
            with log.open('wb') as f:
                rc=subprocess.call([PY,str(ROOT/'scripts/r3_sim2sim_completed_reference_audit.py')],cwd=ROOT,env=environment(),stdout=f,stderr=subprocess.STDOUT)
            assert rc==0, 'Reference numerical audit failed: '+str(log)
        (MON/(PREFIX+'.state')).write_text('COMPLETE requested_stage='+a.stage+' tasks='+','.join(tasks)+'\n')

if __name__=='__main__':main()
