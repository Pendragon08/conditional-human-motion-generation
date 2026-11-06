"""Lossless, lease-protected archival of successful paper_v11 cohort logs only."""
import argparse
import errno
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'repair_prior29_v2_20261105'
BANK = STAGE / 'new_references_paper_v11'
OUT = STAGE / 'completed_cohort_log_archive_v1'
TASKS = ('pick', 'pour', 'bimanual_box_lift', 'box_push_pull')
FLOOR = 8 * 1024**3 + 456 * 1024**2
BROKEN = False


def on_break(*_):
    global BROKEN
    BROKEN = True


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(s):
    return {k: str(getattr(s, k)) for k in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_mode', 'st_nlink')}


def digest_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    h = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024**2)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()


def digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
        return h.hexdigest()


def completion(cohort):
    proof = {}
    result = json.loads((cohort / 'result.json').read_text())
    assert result['status'] == 'cohort_review_completed' and result['native_pass'] > 0
    assert result['task'] == cohort.parent.name and result['cohort'] == int(cohort.name.split('_')[1])
    proof['result_sha256'] = digest(cohort / 'result.json')
    for phase in ('sample', 'postopt', 'native'):
        path = cohort / (phase + '_exit.json')
        assert json.loads(path.read_text())['returncode'] == 0
        proof[phase + '_exit_sha256'] = digest(path)
    return proof


def archive_one(source, bank, execute=False):
    global BROKEN
    assert source.name in ('sample.log', 'postopt.log', 'native.log')
    assert source.parent.parent.name in TASKS and source.parent.name.startswith('cohort_')
    assert source.parent.parent.parent == bank and source.resolve(strict=True) == source
    assert bank.resolve(strict=True) == bank
    proof = completion(source.parent)
    before = os.stat(source, follow_symlinks=False)
    assert stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
    archive = source.with_name(source.name + '.gz')
    partial = source.with_name(source.name + '.gz.partial')
    marker = source.with_name(source.name + '.archived.json')
    assert not any(x.exists() for x in (archive, partial, marker)), 'Existing archive or evidence requires review'
    if not execute:
        return {'source': str(source), 'bytes': before.st_size, 'completion': proof}
    assert shutil.disk_usage(source.parent).free > FLOOR + before.st_size + 20 * 1024**2
    src_fd = os.open(source, os.O_RDWR | os.O_NOFOLLOW)
    dst_fd = None
    try:
        BROKEN = False
        try:
            fcntl.fcntl(src_fd, fcntl.F_SETLEASE, fcntl.F_WRLCK)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                return {'source': str(source), 'status': 'skipped_open_source', 'source_removed': False}
            raise
        assert identity(os.fstat(src_fd)) == identity(before)
        source_sha = digest_fd(src_fd)
        dst_fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        fcntl.fcntl(dst_fd, fcntl.F_SETLEASE, fcntl.F_WRLCK)
        os.lseek(src_fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(dst_fd), 'wb') as out:
            with gzip.GzipFile(fileobj=out, mode='wb', compresslevel=3, mtime=0, filename='') as gz:
                while True:
                    chunk = os.read(src_fd, 1024**2)
                    if not chunk:
                        break
                    assert not BROKEN
                    gz.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        archive_size = os.fstat(dst_fd).st_size
        archive_sha = digest_fd(dst_fd)
        passes = []
        for number in (1, 2):
            assert digest_fd(dst_fd) == archive_sha
            os.lseek(dst_fd, 0, os.SEEK_SET)
            h, count = hashlib.sha256(), 0
            with os.fdopen(os.dup(dst_fd), 'rb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='rb') as gz:
                    while True:
                        chunk = gz.read(1024**2)
                        if not chunk:
                            break
                        h.update(chunk)
                        count += len(chunk)
            assert h.hexdigest() == source_sha and count == before.st_size
            passes.append({'pass': number, 'full_gzip_sha256': archive_sha,
                           'full_decompressed_sha256': h.hexdigest(), 'decompressed_bytes': count})
        assert archive_size < before.st_size, 'Compression did not save space; source preserved'
        assert completion(source.parent) == proof
        assert digest_fd(src_fd) == source_sha
        assert identity(os.stat(source, follow_symlinks=False)) == identity(os.fstat(src_fd)) == identity(before)
        assert not BROKEN
        assert all(fcntl.fcntl(fd, fcntl.F_GETLEASE) == fcntl.F_WRLCK for fd in (src_fd, dst_fd))
        os.link(partial, archive, follow_symlinks=False)
        os.unlink(partial)
        entry = {'source': str(source), 'archive': str(archive), 'source_identity': identity(before),
                 'source_sha256': source_sha, 'source_bytes': before.st_size, 'gzip_sha256': archive_sha,
                 'gzip_bytes': archive_size, 'verification_passes': passes, 'completion': proof,
                 'exclusive_source_and_archive_kernel_leases': True, 'source_removed': False,
                 'restore': 'Decompress archive to a NEW temporary file; verify source SHA and size; create original path exclusively; preserve archive and refuse differing content.'}
        save(marker, entry)
        assert digest_fd(dst_fd) == archive_sha and digest_fd(src_fd) == source_sha
        assert identity(os.stat(source, follow_symlinks=False)) == identity(os.fstat(src_fd)) == identity(before)
        assert not BROKEN
        assert all(fcntl.fcntl(fd, fcntl.F_GETLEASE) == fcntl.F_WRLCK for fd in (src_fd, dst_fd))
        os.unlink(source)
        entry.update(source_removed=True, released_unix=time.time(), saved_bytes=before.st_size - archive_size)
        save(marker, entry)
        return entry
    finally:
        if dst_fd is not None:
            os.close(dst_fd)
        os.close(src_fd)


def self_test():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='self_test_', dir=OUT) as temp:
        bank = Path(temp)
        cohort = bank / 'pick/cohort_9999'
        cohort.mkdir(parents=True)
        save(cohort / 'result.json', {'status': 'cohort_review_completed', 'native_pass': 1, 'task': 'pick', 'cohort': 9999})
        for phase in ('sample', 'postopt', 'native'):
            save(cohort / (phase + '_exit.json'), {'returncode': 0})
        source = cohort / 'postopt.log'
        content = (b'complete immutable test log\n' * 20000)
        source.write_bytes(content)
        child = subprocess.Popen([sys.executable, '-c',
            'import sys; f=open(sys.argv[1]); print("ready",flush=True); sys.stdin.read(1)', str(source)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == 'ready'
            blocked = archive_one(source, bank, True)
            assert blocked['status'] == 'skipped_open_source' and source.read_bytes() == content
        finally:
            child.communicate('x', timeout=5)
        receipt = archive_one(source, bank, True)
        assert receipt['source_removed'] and not source.exists()
        assert gzip.decompress(Path(receipt['archive']).read_bytes()) == content
        source.write_bytes(content)
        try:
            archive_one(source, bank, True)
            raise RuntimeError('Archive collision was not rejected')
        except AssertionError as exc:
            assert 'Existing archive' in str(exc)
        assert source.read_bytes() == content
        failed = cohort / 'native.log'
        failed.write_bytes(content)
        save(cohort / 'native_exit.json', {'returncode': 1})
        try:
            archive_one(failed, bank, True)
            raise RuntimeError('Failed phase was not rejected')
        except AssertionError:
            pass
        assert failed.read_bytes() == content
    result = {'status': 'passed', 'checks': ['active_open_rejected', 'two_complete_readbacks',
        'exact_restore_bytes', 'existing_archive_refused', 'failed_phase_preserved'], 'source_sha256': digest(Path(__file__))}
    save(OUT / 'self_test.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGIO, on_break)
    if args.self_test:
        return self_test()
    OUT.mkdir(exist_ok=True)
    with (OUT / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.execute:
            candidates = []
            for task in TASKS:
                for source in sorted((BANK / task).glob('cohort_*/*.log')):
                    if source.stat().st_size < 256 * 1024 or not (source.parent / 'result.json').exists():
                        continue
                    try:
                        candidates.append(archive_one(source, BANK))
                    except (AssertionError, FileNotFoundError):
                        continue
            print(json.dumps({'mode': 'check', 'candidates': candidates}), flush=True)
            return
        state = {'pid': os.getpid(), 'status': 'watching_completed_cohorts', 'source_sha256': digest(Path(__file__)),
                 'bank': str(BANK), 'archived_files': 0, 'saved_bytes': 0}
        try:
            assert not (OUT / 'status.json').exists(), 'Existing execution status requires review'
            while True:
                state.update(updated_unix=time.time(), free_data_bytes=shutil.disk_usage(ROOT).free)
                save(OUT / 'status.json', state)
                for task in TASKS:
                    for source in sorted((BANK / task).glob('cohort_*/*.log')):
                        if source.stat().st_size < 256 * 1024 or not (source.parent / 'result.json').exists():
                            continue
                        try:
                            completion(source.parent)
                        except (AssertionError, FileNotFoundError):
                            continue
                        if shutil.disk_usage(ROOT).free >= 12 * 1024**3:
                            continue
                        receipt = archive_one(source, BANK, True)
                        if receipt.get('source_removed'):
                            with (OUT / 'manifest.jsonl').open('a') as f:
                                f.write(json.dumps(receipt) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                            state['archived_files'] += 1
                            state['saved_bytes'] += receipt['saved_bytes']
                            save(OUT / 'status.json', state)
                parent = json.loads((BANK / 'status.json').read_text())
                state['generation_status'] = parent['status']
                if not args.watch or parent['status'] in ('failed_preserved', 'ready_for_eval'):
                    state.update(status='finished', updated_unix=time.time())
                    save(OUT / 'status.json', state)
                    break
                time.sleep(30)
        except BaseException as exc:
            state.update(status='failed_preserved', error=repr(exc), traceback=traceback.format_exc())
            save(OUT / 'status.json', state)
            raise


if __name__ == '__main__':
    main()
