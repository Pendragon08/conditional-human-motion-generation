"""Independently verify the previously exported mixed PPO 1000 actor on CPU."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_augmentation2526_mixed1000_cpu_verify_v1'
EXPORT = 'user_augmentation2526_mixed1000_cpu_export_v1'
SUPERVISOR = ROOT / 'scripts/supervise_augmentation2526_mixed1000_cpu_verify_v1.py'


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--trusted-request-sha256', required=True)
    args = p.parse_args()
    request = pathlib.Path(args.request)
    if not __debug__ or os.getuid() != 1013 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('invalid CPU environment')
    if request.is_symlink() or digest(request) != args.trusted_request_sha256:
        raise RuntimeError('request mismatch')
    job = json.loads(request.read_text())
    if job['run'] != RUN or job['role'] != 'verify' or job['authorized'] is not True or digest(__file__) != job['executor_sha256']:
        raise RuntimeError('wrong verification job')
    required = ('augmentation5368_audit_binding_v1', 'augmentation2526_mixed1000_training_acceptance_v1',
                'augmentation2526_stable_file_v1', 'filtered15000_vae_completion',
                'augmentation2526_mixed_ppo_runtime_v1', 'augmentation2526_mixed1000_cpu_core_v1',
                'augmentation2526_mixed1000_fullstate_metadata_v1', 'augmentation2526_mixed1000_cpu_verify_core_v1',
                'supervise_augmentation2526_mixed1000_cpu_verify_v1')
    if not {'scripts/' + n + '.py' for n in required}.issubset(job['bound_sources']):
        raise RuntimeError('missing source binding')
    for name, expected in job['bound_sources'].items():
        path = ROOT / name
        if not name.startswith('scripts/') or '..' in pathlib.PurePosixPath(name).parts or path.resolve() != path or path.stat().st_uid != 1013 or digest(path) != expected:
            raise RuntimeError('source mismatch')
    from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
    pid = os.getppid()
    for _ in range(8):
        if pid <= 1: break
        proc = pathlib.Path('/proc') / str(pid)
        argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        if proc.stat().st_uid == 1013 and str(SUPERVISOR) in argv and str(request) in argv:
            st = (ROOT / 'receipts/full_library_capacity.lock').stat()
            if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(), pid, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                raise RuntimeError('common lock not held by supervisor')
            break
        pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    else:
        raise RuntimeError('supervisor ancestry not found')
    if pid <= 1: raise RuntimeError('supervisor ancestry not found')
    from augmentation2526_mixed1000_training_acceptance_v1 import accepted_mixed_training
    training = accepted_mixed_training(job['training_binding'])['binding']
    parent = ROOT / 'step34_20261029'
    paths = dict(request=ROOT/'receipts'/(EXPORT+'_request.json'), driver=ROOT/'receipts'/(EXPORT+'_driver.json'),
                 outer=ROOT/'receipts'/(EXPORT+'_outer_exit.json'), receipt=parent/EXPORT/'receipt.json',
                 exit=parent/(EXPORT+'_supervision')/'exit.json', process=parent/(EXPORT+'_supervision')/'process_exit.json',
                 resources=parent/(EXPORT+'_supervision')/'resources.jsonl', actor=parent/EXPORT/'actor_RMS.private.pth')
    if set(job['export_hashes']) != set(paths): raise RuntimeError('incomplete export binding')
    for key, path in paths.items():
        if path.resolve() != path or path.stat().st_uid != 1013 or digest(path) != job['export_hashes'][key]:
            raise RuntimeError('export evidence mismatch: ' + key)
    export = json.loads(paths['receipt'].read_text())
    if shutil.disk_usage(parent).free < 2*1024**3+128*1024**2+32*1024**2:
        raise RuntimeError('insufficient output reserve')
    out = parent / RUN
    out.mkdir(mode=0o700, exist_ok=False)
    rec = dict(status='VERIFYING_MIXED_PPO1000_ACTOR', output_writers_closed=False,
               request_sha256=digest(request), export_hashes=job['export_hashes'], training_binding=training)
    def persist():
        tmp=out/'receipt.partial'
        with tmp.open('w') as f:
            json.dump(rec,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        tmp.replace(out/'receipt.json')
    persist()
    try:
        from augmentation2526_mixed1000_cpu_verify_core_v1 import verify
        result = verify(training['checkpoint']['path'], paths['actor'], training, export)
        rec.update(status='MIXED_PPO1000_ACTOR_INDEPENDENT_CPU_VERIFY_FINISHED',
                   output_writers_closed=True, independent_process_verification=True, result=result)
        persist()
    except BaseException as error:
        rec.update(status='MIXED_PPO1000_INDEPENDENT_VERIFY_FAILED', error=repr(error), traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__': main()
