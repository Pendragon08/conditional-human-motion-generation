"""Separate accepted resumed TMR100 immutable-byte backup; no model deserialization."""
import argparse
import hashlib
import json
import msvcrt
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
from backup_filter_recovery_checkpoint import recorded_remote
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/semantic_evaluator_20261029'
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
P = D / 'checkpoints/private/tmr100_endpoint_20261029'
sys.path.insert(0, str(B / 'scripts'))
from tmr100_backup_contract import validate_descriptor
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--worker', action='store_true')
p.add_argument('--transfer-profile', choices=['default', 'conservative'], default='default')
a = p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
E = P / 'transfers' / a.run


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def birth(process):
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.GetProcessTimes.restype = wintypes.BOOL
    ts = [wintypes.FILETIME() for _ in range(4)]
    assert k.GetProcessTimes(wintypes.HANDLE(int(process._handle)), *[ctypes.byref(x) for x in ts])
    return (ts[0].dwHighDateTime << 32) | ts[0].dwLowDateTime


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


locks = []


def acquire_locks():
    # Share the existing VAE transport lock as well as the LDM and TMR locks.
    # This never takes the GPU/model lock and cannot stop a numerical job.
    for name in ['filtered15000_vae_checkpoint_backup.lock', 'ldm100k_checkpoint_backup.lock', 'tmr100_checkpoint_backup.lock']:
        guard_local(D, D / name)
        f = (D / name).open('a+b')
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except BaseException:
            f.close()
            raise
        locks.append(f)


if not a.worker:
    old_copy = B / 'results/vae100k_user_handoff_20261028/r997'
    if (old_copy / 's_client_intent.json').exists():
        outcome = old_copy / 's_client.json'
        if not outcome.exists() or not json.loads(outcome.read_bytes()).get('client_reaped'):
            print(json.dumps(dict(status='ORIGINAL_PRIVATE_CLIENT_UNKNOWN_NO_DISPATCH')))
            raise SystemExit(0)
    roots = [B / 'results/vae100k_user_handoff_20261028', D / 'checkpoints/private/ldm100k_endpoint_20261028/transfers', P / 'transfers']
    for root in roots:
        for intent in root.glob('*/client_intent.json'):
            outcome, end = intent.with_name('client_outcome.json'), intent.with_name('worker_exit.json')
            if not outcome.exists() and not (end.exists() and json.loads(end.read_bytes()).get('client_reaped')):
                print(json.dumps(dict(status='EXISTING_PRIVATE_TRANSPORT_UNKNOWN_NO_DISPATCH', intent=str(intent))))
                raise SystemExit(0)
    try:
        acquire_locks()
    except OSError:
        for f in locks:
            f.close()
        print(json.dumps(dict(status='PRIVATE_TRANSPORT_LOCK_BUSY_NO_DISPATCH')))
        raise SystemExit(0)
    for f in locks:
        f.close()
    j = json.loads((O / 'tmr100_frozen_export_v1/progress.json').read_bytes())
    assert j['closed_execution_verified'] and not j['workers']
    rec = j['files']['receipt']
    assert rec['status'] == 'TMR100_TRAINED_FROZEN_CPU_EXPORTED'
    descriptor = dict(checkpoint=rec['training_binding']['selected_checkpoint'],
        training_binding=rec['training_binding'], export_binding=j['artifact_audit']['export_binding'])
    validate_descriptor(descriptor['checkpoint'], descriptor['export_binding'], descriptor['training_binding'])
    assert not E.exists() and not (O / a.run).exists(), 'Existing copy run/intent must be reconciled'
    guard_local(D, P, directory=True)
    E.mkdir(parents=True, exist_ok=False)
    write(E / 'source_binding.json', descriptor)
    cmd = [sys.executable, '-B', str(pathlib.Path(__file__).resolve()), '--run', a.run,
           '--transfer-profile', a.transfer_profile, '--worker']
    with (E / 'worker.log').open('xb') as log:
        q = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    result = dict(status='PRIVATE_TMR100_COPY_DISPATCHED_NOT_VERIFIED', worker_pid=q.pid,
                   worker_start_filetime=birth(q), command=cmd, D_evidence_dir=str(E),
                   transfer_profile=a.transfer_profile, started_unix=time.time())
    write(E / 'dispatch.json', result)
    local = O / a.run
    local.mkdir(exist_ok=False)
    write(local / 'dispatch.json', result)
    print(json.dumps(result))
    raise SystemExit(0)

client = None
try:
    acquire_locks()
    descriptor = json.loads((E / 'source_binding.json').read_bytes())
    ck, binding, training = descriptor['checkpoint'], descriptor['export_binding'], descriptor['training_binding']
    validate_descriptor(ck, binding, training)
    dest = P / 'epoch_100'
    guard_local(D, dest, directory=True)
    dest.mkdir(exist_ok=True)
    final, partial = dest / 'tmr_epoch_100_resumed.pth', dest / 'tmr_epoch_100_resumed.pth.partial'
    if final.exists():
        guard_local(D, final)
        assert final.stat().st_size == ck['bytes'] and digest(final) == ck['sha256']
        assert shutil.disk_usage(D).free >= 2 * 1024**3
        write(E / 'backup.json', dict(status='USER_TMR100_PRIVATE_FULLSTATE_BYTE_BACKUP_VERIFIED',
              checkpoint=ck, D_path=str(final), D_sha256=ck['sha256'], verified_existing_copy=True,
              new_transfer_dispatched=False, D_free_bytes=shutil.disk_usage(D).free))
        write(E / 'worker_exit.json', dict(actual_returncode=0, status='EXACT_PRIVATE_COPY_ALREADY_PRESENT'))
        raise SystemExit(0)
    guard_local(D, partial)
    count = partial.stat().st_size if partial.exists() else 0
    assert 0 <= count <= ck['bytes']
    assert shutil.disk_usage(D).free >= 2 * 1024**3 + ck['bytes'] - count + 64 * 1024**2
    prefix_sha = digest(partial) if count else hashlib.sha256(b'').hexdigest()
    code = r'''
import hashlib,json,os,pathlib,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from tmr100_trained_acceptance import accepted_frozen_export,fingerprint,sha
ck=__CHECKPOINT__;binding=__EXPORT_BINDING__;training=__TRAINING_BINDING__
expected=__HELPER_SHA__
for name,value in expected.items():assert sha(r/name)==value,name
v=accepted_frozen_export(binding);assert v['training']['binding']==training
assert training['kind']=='USER_TMR100_CAPACITY_RESUME_ORIGINAL_FAILURE_RETAINED'
assert training['selected_checkpoint']==ck and ck['epoch']==100 and ck['optimizer_updates']==101500
assert ck['path']=='/home/ouyanghao/textop-tmr69d-evaluator-20261029/resume_checkpoints_v1/tmr_epoch_100_resumed.pth'
source_fingerprint=fingerprint(ck['path'],home=True)
assert {k:source_fingerprint[k] for k in ['path','bytes','sha256']}=={k:ck[k] for k in ['path','bytes','sha256']}
left=__PREFIX_BYTES__;h=hashlib.sha256()
with pathlib.Path(ck['path']).open('rb') as f:
 while left:
  b=f.read(min(left,8*1024**2));assert b;left-=len(b);h.update(b)
print(json.dumps(dict(fingerprint=source_fingerprint,prefix_sha256=h.hexdigest())))
'''.replace('__CHECKPOINT__', repr(ck)).replace('__EXPORT_BINDING__', repr(binding)).replace('__PREFIX_BYTES__', repr(count)).replace('__TRAINING_BINDING__', repr(training)).replace('__HELPER_SHA__', repr({n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ['scripts/tmr100_trained_acceptance.py','scripts/tmr100_capacity_resumed_acceptance.py']}))
    before = json.loads(recorded_remote(code, E / 'before_client.json', timeout=180))
    assert before['prefix_sha256'] == prefix_sha
    write(E / 'intent.json', dict(checkpoint=ck, export_binding=binding, training_binding=training, prefix_bytes=count,
          prefix_sha256=prefix_sha, source_fingerprint=before['fingerprint'],
          transfer_profile=a.transfer_profile, new_model_loads=0, new_gpu_contexts=0))
    conservative = a.transfer_profile == 'conservative'
    cmd = ['sftp', '-b', '-', '-o', 'BatchMode=yes', '-o',
           'ConnectTimeout=' + ('30' if conservative else '15'), '-o',
           'ServerAliveInterval=' + ('10' if conservative else '30'),
           '-i', 'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32']
    if conservative:
        cmd[1:1] = ['-R', '1', '-B', '16384', '-o', 'ServerAliveCountMax=12']
    with (E / 'transfer.stdout.txt').open('xb') as out, (E / 'transfer.stderr.txt').open('xb') as err, (E / 'resources.jsonl').open('x', encoding='utf-8') as resources:
        client = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=dest)
        write(E / 'client_intent.json', dict(command=cmd, pid=client.pid, start_filetime=birth(client),
              timeout_seconds=21600, started_unix=time.time(), prefix_bytes=count,
              transfer_profile=a.transfer_profile))
        client.stdin.write(('reget ' + ck['path'] + ' tmr_epoch_100_resumed.pth.partial\n').encode())
        client.stdin.close()
        started = time.monotonic()
        while client.poll() is None:
            size = partial.stat().st_size if partial.exists() else 0
            free = shutil.disk_usage(D).free
            resources.write(json.dumps(dict(unix=time.time(), bytes=size, D_free_bytes=free, client_pid=client.pid)) + '\n')
            resources.flush()
            assert size <= ck['bytes'] and free >= 2 * 1024**3 + max(0, ck['bytes'] - size) + 16 * 1024**2
            assert time.monotonic() - started < 21600
            try:
                client.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        write(E / 'client_outcome.json', dict(actual_client_exit_code=client.returncode,
              client_reaped=True, streams_final=True, prefix_bytes_preserved=count))
        assert client.returncode == 0
    guard_local(D, partial)
    assert partial.stat().st_size == ck['bytes'] and digest(partial) == ck['sha256']
    with partial.open('r+b') as f:
        f.flush()
        os.fsync(f.fileno())
    after = json.loads(recorded_remote(code, E / 'after_client.json', timeout=180))
    assert after == before
    partial.rename(final)
    guard_local(D, final)
    assert shutil.disk_usage(D).free >= 2 * 1024**3
    result = dict(status='USER_TMR100_PRIVATE_FULLSTATE_BYTE_BACKUP_VERIFIED', checkpoint=ck,
        export_binding=binding, training_binding=training, D_path=str(final), D_sha256=ck['sha256'], source_fingerprint_stable=True,
        D_free_bytes=shutil.disk_usage(D).free, actual_sftp_exit_code=0, resumed_prefix_bytes=count,
        new_model_loads=0, new_gpu_contexts=0)
    write(E / 'backup.json', result)
    if not (dest / 'backup.json').exists():
        write(dest / 'backup.json', result)
    write(E / 'worker_exit.json', dict(status='PRIVATE_TMR100_BACKUP_VERIFIED', actual_returncode=0))
except BaseException as error:
    if isinstance(error, SystemExit):
        raise
    if client is not None and client.poll() is None:
        client.kill()  # Only this directly spawned local SFTP client.
        client.wait(timeout=15)
    if client is not None and not (E / 'client_outcome.json').exists():
        write(E / 'client_outcome.json', dict(actual_client_exit_code=client.returncode, client_reaped=True, streams_final=True, local_guard_failed=True))
    write(E / 'worker_exit.json', dict(status='PRIVATE_TMR100_COPY_FAILED_PARTIAL_RETAINED', actual_returncode=1,
          error=repr(error), traceback=traceback.format_exc(), client_reaped=client is None or client.poll() is not None))
    raise
finally:
    for f in locks:
        f.close()
