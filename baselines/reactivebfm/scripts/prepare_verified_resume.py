"""Validate a completed checkpoint pair and write a recovery plan; never launch.

The original training loop can leave incomplete highest-numbered files. Only a
ready manifest whose model AND optimizer match their sizes/SHA256 is eligible.
Recovery must use a fresh output directory so upstream filename scanning cannot
select an unverified partial file. This does not promise exact RNG/data replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


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
            assert type(step) is int and type(u) is int and step>=0 and 0<u<=step+1
            assert manifest.name==f'checkpoint_ready_{step:09d}.json'
            rows=m['files'];assert len(rows)==2
            assert {Path(x['file']).name for x in rows}=={f'model{step:09d}.pt',f'opt{step:09d}.pt'}
            for row in rows:
                p=Path(row['file'])
                assert not p.is_symlink() and p.resolve().parent==root
                assert re.fullmatch('[0-9a-f]{64}',row['sha256'])
                assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--new-output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--plan-output',type=Path,required=True);a=p.parse_args()
    plan=recovery_plan(a.run_dir,a.new_output,a.config)
    with a.plan_output.open('x',encoding='utf-8') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan))
