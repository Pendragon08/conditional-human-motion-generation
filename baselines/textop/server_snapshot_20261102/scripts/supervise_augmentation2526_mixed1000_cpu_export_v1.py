"""Local-only draft for an exclusively locked mixed PPO 1000 CPU export.

Not deployed or ready. A separately trusted launcher must supply an immutable
request digest, authenticate this file before execution, and record its exit.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_augmentation2526_mixed1000_cpu_export_v1'
ENTRY = ROOT / 'scripts/run_augmentation2526_mixed1000_cpu_export_v1.py'
FLOOR = 2 * 1024**3 + 128 * 1024**2 + 32 * 1024**2
REQUIRED = frozenset('scripts/' + n + '.py' for n in (
    'supervise_augmentation2526_mixed1000_cpu_export_v1',
    'run_augmentation2526_mixed1000_cpu_export_v1',
    'augmentation5368_audit_binding_v1', 'augmentation2526_mixed1000_training_acceptance_v1',
    'augmentation2526_stable_file_v1', 'filtered15000_vae_completion',
    'augmentation2526_mixed_ppo_runtime_v1', 'augmentation2526_mixed1000_cpu_core_v1',
    'augmentation2526_mixed1000_fullstate_metadata_v1',
    'contact_filter_resources', 'contact_process_supervision')) | frozenset(('scripts/run_tracker.sh',))


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--trusted-request-sha256', required=True)
    args = parser.parse_args()
    request = pathlib.Path(args.request)
    if not __debug__ or os.getuid() != 1013 or request.is_symlink():
        raise RuntimeError('invalid supervisor identity or request')
    if sha(request) != args.trusted_request_sha256:
        raise RuntimeError('request hash differs from trusted launcher binding')
    job = json.loads(request.read_text())
    if job['run'] != RUN or job['role'] != 'export' or job['authorized'] is not True:
        raise RuntimeError('wrong job')
    if not isinstance(job.get('training_binding'), dict) or not job['training_binding']:
        raise RuntimeError('an explicit completed-training binding is required')
    if sha(__file__) != job['supervisor_sha256'] or sha(ENTRY) != job['executor_sha256']:
        raise RuntimeError('supervisor or entry mismatch')
    if not REQUIRED.issubset(job['bound_sources']):
        raise RuntimeError('missing required source binding')
    for name, expected in job['bound_sources'].items():
        if not name.startswith('scripts/') or '..' in pathlib.PurePosixPath(name).parts:
            raise RuntimeError('invalid source path')
        path = ROOT / name
        if path.resolve() != path or path.stat().st_uid != 1013 or sha(path) != expected:
            raise RuntimeError('source mismatch: ' + name)
    from augmentation2526_mixed1000_training_acceptance_v1 import accepted_mixed_training
    from contact_filter_resources import sample_resources
    from contact_process_supervision import run_supervised
    os.sched_setaffinity(0, {0, 1, 2, 3})
    lock_path = ROOT / 'receipts/full_library_capacity.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        accepted_mixed_training(job['training_binding'])
        parent = ROOT / 'step34_20261029'
        out, log = parent / RUN, parent / (RUN + '_supervision')
        if out.exists() or log.exists() or shutil.disk_usage(parent).free < FLOOR:
            raise RuntimeError('existing intent or insufficient capacity')
        cache = parent / (RUN + '_pycache_disabled')
        if cache.exists() or cache.is_symlink():
            raise RuntimeError('isolated bytecode cache path already exists')
        sample_resources(ROOT, None, 16 * 1024**3, FLOOR)
        log.mkdir(mode=0o700, exist_ok=False)

        def sample(pid):
            if shutil.disk_usage(parent).free < FLOOR:
                raise RuntimeError('output reserve unavailable')
            return sample_resources(ROOT, pid, 16 * 1024**3, FLOOR)

        cmd = ['taskset', '-c', '0-3', '/usr/bin/env', '-u', 'PYTHONPATH',
               '-u', 'PYTHONOPTIMIZE', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
               'PYTHONPYCACHEPREFIX=' + str(cache),
               'PYTHONIOENCODING=utf-8', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
               'bash', str(ROOT / 'scripts/run_tracker.sh'), str(ENTRY),
               '--request', str(request), '--trusted-request-sha256', args.trusted_request_sha256]
        proof = run_supervised(cmd, log / 'child.log', log / 'resources.jsonl',
                               log / 'process_exit.json', sample, 900, 15)
        receipt_path = out / 'receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        proof.update(run=RUN, source_sha256=sha(__file__), request_sha256=sha(request),
                     receipt_sha256=sha(receipt_path) if receipt_path.exists() else None,
                     resource_sha256=sha(log / 'resources.jsonl'),
                     process_exit_sha256=sha(log / 'process_exit.json'))
        with (log / 'exit.json').open('x') as f:
            json.dump(proof, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        if not (proof['supervision_passed'] and receipt.get('output_writers_closed') is True and
                receipt.get('status') == 'MIXED_PPO1000_CPU_EXPORTED_AWAITING_INDEPENDENT_VERIFY'):
            raise RuntimeError('export did not complete; preserve evidence')


if __name__ == '__main__':
    main()
