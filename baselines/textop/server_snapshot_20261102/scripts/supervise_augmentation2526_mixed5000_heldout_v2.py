"""Draft exclusive supervisor for mixed-5000 held-out simulation."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUNS={'pilot':'user_augmentation2526_mixed5000_heldout_pilot_v2',
      'full':'user_augmentation2526_mixed5000_heldout_v2'}
ENTRY=ROOT/'scripts/run_augmentation2526_mixed5000_heldout_v2.py'
REQUIRED={'scripts/'+n for n in (
    'supervise_augmentation2526_mixed5000_heldout_v2.py','run_augmentation2526_mixed5000_heldout_v2.py',
    'augmentation2526_mixed5000_heldout_inputs_v1.py','augmentation2526_mixed5000_heldout_actor_v1.py',
    'augmentation2526_mixed5000_heldout_rollout_v1.py','ppo5000_heldout_manifest_contract.py',
    'ppo5000_heldout_env_preparation.py','ppo5000_heldout_aggregate.py','strict_tracking_coverage.py',
    'direct_strict_trace.py','augmentation5368_audit_binding_v1.py','solid_floor_diagnostic.py',
    'contact_filter_resources.py','contact_process_supervision.py','bounded_headless_close.py','run_tracker.sh')}
TRACKER='upstream/TextOpTracker/source/textop_tracker/textop_tracker/'
REQUIRED|={'configs/tracker-env-spec.json'}|{TRACKER+n for n in (
    'robots/g1.py','tasks/tracking/tracking_env_cfg.py','tasks/tracking/mdp/commands_multi.py',
    'tasks/tracking/mdp/observations.py','tasks/tracking/mdp/rewards.py','tasks/tracking/mdp/terminations.py',
    'tasks/tracking/config/g1/flat_env_cfg.py','tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py')}
FLOOR=2*1024**3+128*1024**2+512*1024**2


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True)
    a=p.parse_args();request=pathlib.Path(a.request)
    if not __debug__ or os.getuid()!=1013 or request.resolve()!=request or sha(request)!=a.trusted_request_sha256:
        raise ValueError('invalid request identity')
    job=json.loads(request.read_text())
    run=RUNS.get(job.get('role'))
    if run is None or job['run']!=run or job['authorized'] is not True or job['stage']!='heldout':raise ValueError('wrong evaluation')
    expected_indices=list(range(0,1880,117))[:16] if job['role']=='pilot' else list(range(1880))
    if job['trial_indices']!=expected_indices:raise ValueError('unexpected trial selection')
    if sha(__file__)!=job['supervisor_sha256'] or sha(ENTRY)!=job['executor_sha256']:
        raise ValueError('entry or supervisor differs')
    if not REQUIRED.issubset(job['bound_sources']):raise ValueError('incomplete source binding')
    for name,digest in job['bound_sources'].items():
        path=ROOT/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=digest:
            raise ValueError('source differs: '+name)
    from contact_filter_resources import sample_resources
    from contact_process_supervision import run_supervised
    from augmentation2526_mixed5000_heldout_inputs_v1 import accept
    os.sched_setaffinity(0,{0,1,2,3})
    parent=ROOT/'step34_20261029';out=parent/run;logs=parent/(run+'_supervision')
    with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists() or logs.exists() or shutil.disk_usage(parent).free<FLOOR:
            raise RuntimeError('existing intent or insufficient disk space')
        accept(job['input_binding'])
        sample_resources(ROOT,None,16*1024**3,FLOOR)
        gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'],text=True).strip())*1024**2
        if gpu<job['gpu_memory_floor_bytes']:raise RuntimeError('insufficient GPU memory')
        cache=parent/(run+'_pycache_disabled')
        if cache.exists() or cache.is_symlink():raise RuntimeError('bytecode cache path exists')
        logs.mkdir(mode=0o700,exist_ok=False)
        def sample(pid):
            if shutil.disk_usage(parent).free<FLOOR:raise RuntimeError('output reserve unavailable')
            return sample_resources(ROOT,pid,16*1024**3,FLOOR)
        cmd=['taskset','-c','0-3','/usr/bin/env','-u','PYTHONPATH','-u','PYTHONOPTIMIZE',
             '-u','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER=PCI_BUS_ID','PYTHONDONTWRITEBYTECODE=1',
             'PYTHONPYCACHEPREFIX='+str(cache),'PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
             'bash',str(ROOT/'scripts/run_tracker.sh'),str(ENTRY),'--request',str(request),
             '--trusted-request-sha256',a.trusted_request_sha256,'--device','cuda:1','--headless']
        proof=run_supervised(cmd,logs/'child.log',logs/'resources.jsonl',logs/'process_exit.json',sample,7200,15)
        recpath=out/'receipt.json';rec=json.loads(recpath.read_text()) if recpath.exists() else {}
        proof.update(run=run,request_sha256=sha(request),receipt_sha256=sha(recpath) if recpath.exists() else None,
                     resource_sha256=sha(logs/'resources.jsonl'),process_exit_sha256=sha(logs/'process_exit.json'))
        with (logs/'exit.json').open('x') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        if not (proof['supervision_passed'] and rec.get('status')=='MIXED5000_GENERATED_HELDOUT_PHYSICS_FINISHED'
                and rec.get('output_writers_closed') is True and rec.get('finished_trials')==len(expected_indices) and rec.get('optimizer_updates')==0):
            raise RuntimeError('evaluation did not finish; retain all evidence')


if __name__=='__main__':main()
