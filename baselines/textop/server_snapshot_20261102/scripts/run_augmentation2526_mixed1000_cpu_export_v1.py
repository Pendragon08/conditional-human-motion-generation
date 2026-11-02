"""Local draft for the mixed PPO 1000 CPU audit and actor export.

Not deployed or authorized. The future launcher must authenticate the request
hash and source bundle, hold the common lock in the actual supervisor, enforce
resource limits, and retain all process exits. This is not a policy evaluation.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_augmentation2526_mixed1000_cpu_export_v1'
SUPERVISOR = ROOT / 'scripts/supervise_augmentation2526_mixed1000_cpu_export_v1.py'
RESERVE = 2 * 1024**3
LOG_BUDGET = 128 * 1024**2
OUTPUT_BUDGET = 32 * 1024**2
REQUIRED_SOURCES = frozenset('scripts/' + name + '.py' for name in (
    'supervise_augmentation2526_mixed1000_cpu_export_v1',
    'augmentation5368_audit_binding_v1', 'augmentation2526_mixed1000_training_acceptance_v1',
    'augmentation2526_stable_file_v1', 'filtered15000_vae_completion',
    'augmentation2526_mixed_ppo_runtime_v1', 'augmentation2526_mixed1000_cpu_core_v1',
    'augmentation2526_mixed1000_fullstate_metadata_v1'))


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_supervisor(request):
    from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
    pid = os.getppid()
    for _ in range(8):
        if pid <= 1:
            break
        proc = pathlib.Path('/proc') / str(pid)
        argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        if proc.stat().st_uid == 1013 and str(SUPERVISOR) in argv and str(request) in argv:
            lock = ROOT / 'receipts/full_library_capacity.lock'
            st = lock.stat()
            if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),
                    pid, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                raise RuntimeError('expected supervisor does not hold the common lock')
            return
        pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    raise RuntimeError('expected lock-holding supervisor not found')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--trusted-request-sha256', required=True)
    args = p.parse_args()
    if not __debug__ or os.getuid() != 1013 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('invalid CPU execution environment')
    request = pathlib.Path(args.request)
    if request.is_symlink() or digest(request) != args.trusted_request_sha256:
        raise RuntimeError('request authentication failed')
    job = json.loads(request.read_text())
    if job['run'] != RUN or job['role'] != 'export' or job['authorized'] is not True:
        raise RuntimeError('unexpected job')
    if digest(__file__) != job['executor_sha256']:
        raise RuntimeError('entry source mismatch')
    if not REQUIRED_SOURCES.issubset(job['bound_sources']):
        raise RuntimeError('required source binding missing')
    for name, expected in job['bound_sources'].items():
        if not name.startswith('scripts/') or '..' in pathlib.PurePosixPath(name).parts:
            raise RuntimeError('invalid source path')
        path = ROOT / name
        if path.resolve() != path or path.stat().st_uid != 1013 or digest(path) != expected:
            raise RuntimeError('source mismatch: ' + name)
    require_supervisor(request)
    from augmentation2526_mixed1000_training_acceptance_v1 import accepted_mixed_training
    from augmentation2526_stable_file_v1 import stable_owned_sha256
    training = accepted_mixed_training(job['training_binding'])
    cp = training['binding']['checkpoint']
    if pathlib.Path(cp['path']).name != 'completed_1000.private.pth':
        raise RuntimeError('incorrect checkpoint selected')
    parent = ROOT / 'step34_20261029'
    if shutil.disk_usage(parent).free < RESERVE + LOG_BUDGET + OUTPUT_BUDGET:
        raise RuntimeError('insufficient output capacity')
    out = parent / RUN
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    rec = dict(status='AUDITING_MIXED_PPO1000_ON_CPU', run=RUN, role='export',
               request_sha256=digest(request), training_binding=training['binding'],
               output_writers_closed=False, optimizer_updates_performed=0,
               heldout_tracking_accepted=False, online_latency_accepted=False)

    def persist():
        rec['elapsed_seconds'] = time.monotonic() - started
        tmp = out / 'receipt.partial'
        with tmp.open('w') as f:
            json.dump(rec, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out / 'receipt.json')

    persist()
    try:
        before = stable_owned_sha256(cp['path'], 1013)
        if before['sha256'] != cp['sha256'] or before['bytes'] != cp['bytes']:
            raise RuntimeError('checkpoint bytes differ from training receipt')
        import torch
        from augmentation2526_mixed_ppo_runtime_v1 import content, equal
        from augmentation2526_mixed1000_cpu_core_v1 import audit_and_payload, independent_reload
        if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
            raise RuntimeError('CUDA must remain disabled')
        state = torch.load(cp['path'], map_location='cpu', weights_only=False)
        if content(state) != cp['content_sha256']:
            raise RuntimeError('checkpoint content differs from training receipt')
        payload, audit, expected, probes = audit_and_payload(state, training['binding'])
        partial, final = out / 'actor_RMS.private.pth.partial', out / 'actor_RMS.private.pth'
        with partial.open('xb') as f:
            os.chmod(partial, 0o600)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if not 0 < partial.stat().st_size <= OUTPUT_BUDGET:
            raise RuntimeError('actor export exceeded budget')
        loaded = torch.load(partial, map_location='cpu', weights_only=False)
        if not equal(loaded, payload):
            raise RuntimeError('actor export reload differs')
        probe = independent_reload(loaded, expected, probes)
        after = stable_owned_sha256(cp['path'], 1013)
        if before != after or content(state) != cp['content_sha256']:
            raise RuntimeError('source checkpoint changed during audit')
        if shutil.disk_usage(parent).free < RESERVE + LOG_BUDGET:
            raise RuntimeError('disk reserve lost before finalizing')
        partial.rename(final)
        fd = os.open(str(out), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        artifact = stable_owned_sha256(final, 1013)
        rec.update(status='MIXED_PPO1000_CPU_EXPORTED_AWAITING_INDEPENDENT_VERIFY',
                   output_writers_closed=True, checkpoint_before=before,
                   checkpoint_after=after, inference=artifact, audit=audit, probe=probe,
                   independent_process_verification=False, synthetic_probe_only=True)
        persist()
    except BaseException as error:
        rec.update(status='MIXED_PPO1000_CPU_EXPORT_FAILED_OUTPUTS_RETAINED',
                   error=repr(error), traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__':
    main()
