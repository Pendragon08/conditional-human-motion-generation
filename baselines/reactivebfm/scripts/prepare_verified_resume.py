"""Validate a completed checkpoint pair and write a recovery plan by default.

The original training loop can leave incomplete highest-numbered files. Only a
ready manifest whose model AND optimizer match their sizes/SHA256 is eligible.
Recovery must use a fresh output directory so upstream filename scanning cannot
select an unverified partial file. Optional --execute-plan revalidates at launch
and refuses an active original job. No exact RNG/data replay is promised.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import os
import subprocess

def require(condition,message):
    if not condition:raise ValueError(message)

if not __debug__:
    raise RuntimeError('Optimized Python is unsupported: recovery validation guards must remain enabled')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def select_verified(run):
    run=Path(run).resolve();root=(run/'checkpoints').resolve()
    valid=[];rejected=[]
    for manifest in sorted(run.glob('checkpoint_ready_*.json')):
        try:
            m=json.loads(manifest.read_text(encoding='utf-8'));step=m['official_step'];u=m['optimizer_updates']
            require(type(step) is int and type(u) is int and step>=0 and 0<u<=step+1,'Invalid step/update count')
            require(manifest.name==f'checkpoint_ready_{step:09d}.json','Manifest name mismatch')
            rows=m['files'];require(len(rows)==2,'Model and optimizer are both required')
            require({Path(x['file']).name for x in rows}=={f'model{step:09d}.pt',f'opt{step:09d}.pt'},'Checkpoint pair names mismatch')
            for row in rows:
                p=Path(row['file'])
                require(not p.is_symlink() and p.resolve().parent==root,'Checkpoint escaped expected directory')
                require(bool(re.fullmatch('[0-9a-f]{64}',row['sha256'])),'Invalid SHA256')
                require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Checkpoint size/hash mismatch')
            valid.append({'manifest':str(manifest),'manifest_sha256':sha(manifest),**m})
        except (AssertionError,ValueError,KeyError,TypeError,OSError) as exc:
            rejected.append({'manifest':str(manifest),'error':type(exc).__name__+': '+str(exc)})
    if not valid: raise RuntimeError('No size/SHA256-verified complete checkpoint pair: '+json.dumps(rejected))
    return max(valid,key=lambda m:(m['optimizer_updates'],m['official_step'])),rejected


def recovery_plan(run,new_output,config):
    destination=Path(new_output).resolve()
    if destination.exists():raise FileExistsError('Recovery requires an absent output directory: '+str(destination))
    selected,rejected=select_verified(run)
    model=next(x['file'] for x in selected['files'] if Path(x['file']).name.startswith('model'))
    args=Path(model).parent/'args.json'
    if not args.is_file():raise FileNotFoundError('Original checkpoint args.json is required: '+str(args))
    config=Path(config).resolve()
    if selected.get('configuration_sha256') != sha(config):
        raise ValueError('Recovery configuration differs from the checkpoint manifest')
    return {'status':'VERIFIED_RECOVERY_PLAN_ONLY','checkpoint':selected,'rejected_manifests':rejected,
        'original_run_dir':str(Path(run).resolve()),'config_path':str(config),
        'training_script_sha256':sha(Path(__file__).with_name('train_full.py')),
        'args_sha256':sha(args),'configuration_sha256':sha(config),'new_output':str(destination),
        'command':[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc_per_node=2',str(Path(__file__).with_name('train_full.py')),'--config',str(config),
            '--output',str(destination),'--resume',model],
        'environment':{'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
            'OPENBLAS_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'TOKENIZERS_PARALLELISM':'false'},
        'operator_requirements':['Confirm the prior owned training processes have exited; do not interrupt other jobs.',
            'Use a fresh worker local_output/server_output/evaluation_root and a new Release tag for recovery, preserving the original attempt.',
            'Preserve launch stdout/stderr and restart the backup worker for this new attempt.'],
        'limitations':'No job was launched; model/optimizer recovery is validated, not exact RNG or data-iterator replay.'}


def validate_recovery_plan(plan):
    """Recompute the pair/manifest/args/config/source checks at launch time."""
    current=recovery_plan(plan['original_run_dir'],plan['new_output'],plan['config_path'])
    for key in ['args_sha256','configuration_sha256','training_script_sha256']:
        require(current[key]==plan[key],'Plan-to-launch bytes changed: '+key)
    require(current['checkpoint']['manifest_sha256']==plan['checkpoint']['manifest_sha256'],'Selected manifest changed after planning')
    return current


def execute_plan(plan):
    if not sys.platform.startswith('linux'):raise RuntimeError('GPU recovery execution is supported only on the original Linux server')
    # Do not start a second copy of this run. No process is killed.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\x00')
            if '--output' in args and args[args.index('--output')+1]==plan['original_run_dir']:
                if any(x.endswith('/train_full.py') for x in args):raise RuntimeError('Original training process still running')
        except (OSError,UnicodeError,IndexError):continue
    current=validate_recovery_plan(plan)
    destination=Path(current['new_output']);destination.mkdir(exist_ok=False)
    record={**current,'status':'VERIFIED_RECOVERY_LAUNCH_REQUESTED',
        'limitations':'Pair, manifest, args, config and source revalidated before this launch request; no exact RNG/data replay or successful resumed optimization inferred.'}
    (destination/'verified_recovery_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    environment=os.environ.copy();environment.update(current['environment'])
    return subprocess.run(current['command'],env=environment).returncode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path)
    p.add_argument('--new-output',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--plan-output',type=Path);p.add_argument('--execute-plan',type=Path);a=p.parse_args()
    if a.execute_plan:
        if any([a.run_dir,a.new_output,a.config,a.plan_output]):p.error('Use execute-plan separately from plan creation')
        raise SystemExit(execute_plan(json.loads(a.execute_plan.read_text(encoding='utf-8'))))
    if not all([a.run_dir,a.new_output,a.config,a.plan_output]):p.error('Plan creation requires run-dir, new-output, config and plan-output')
    plan=recovery_plan(a.run_dir,a.new_output,a.config)
    with a.plan_output.open('x',encoding='utf-8') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan))
