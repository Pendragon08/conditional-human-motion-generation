"""Exact private recovery and bounded reclamation of two idle cache roots.

The archive and path manifest stay private. No recursive removal or extraction
is used. An existing different file is never replaced by recovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import stat
import tarfile

ROOTS = [pathlib.Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/cache'),
         pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/cache/downloads')]
ARCHIVE_SHA = '8f6e28ff7d22fad625065c194ed99f6922599ee23a6711c4baad1f2b059e792d'
MANIFEST_SHA = '2ced1247a734ac2bc169b68eb92e5ba1d9eff5cafc5e61a58b46d9d3d4cc1aab'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            s.st_uid, stat.S_IMODE(s.st_mode)]


def safe_parents(path):
    assert path.is_absolute()
    for parent in [path] + list(path.parents):
        if parent.exists() or parent.is_symlink():
            assert not parent.is_symlink(), str(parent)


def persist(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def idle():
    refs, limits = [], []
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != 1013:
                continue
            argv = (proc / 'cmdline').read_bytes()
            name = pathlib.Path(argv.split(b'\0')[0].decode(errors='replace')).name
            for root in ROOTS:
                if str(root).encode() in argv:
                    refs.append([proc.name, 'argv'])
            try:
                maps = (proc / 'maps').read_bytes()
            except PermissionError:
                if name.startswith('sshd:'):
                    name = 'sshd'
                assert name in ['sshd', 'systemd', 'dbus-daemon', 'gpg-agent',
                                'gnome-keyring-daemon', 'ssh-agent', '(sd-pam)'], (proc.name, name)
                limits.append(dict(pid=int(proc.name), service=name))
                continue
            for root in ROOTS:
                if str(root).encode() in maps:
                    refs.append([proc.name, 'maps'])
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    for root in ROOTS:
                        if target == str(root) or target.startswith(str(root) + '/'):
                            refs.append([proc.name, 'fd'])
                except (FileNotFoundError, PermissionError, OSError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert not refs, refs
    return limits


def verified_manifest(path):
    safe_parents(path)
    assert sha(path) == MANIFEST_SHA
    value = json.loads(path.read_bytes())
    assert value['roots'] == [str(p) for p in ROOTS]
    rows = value['entries']
    assert len(rows) == 1000 and sum(x['kind'] == 'file' for x in rows) == 549
    assert len({x['path'] for x in rows}) == len(rows)
    for row in rows:
        p = pathlib.Path(row['path'])
        i = next(i for i, root in enumerate(ROOTS) if p == root or root in p.parents)
        assert '..' not in p.parts and row['member'] == 'root' + str(i) + '/' + str(p.relative_to(ROOTS[i]))
        assert row['kind'] in ['file', 'directory', 'symlink']
        if row['kind'] == 'symlink':
            assert row['member'] == 'root0/xdg/ov/texturecache'
            assert row['target'] == '/dev/shm/ouyanghao_phygile_render_cache_20261029/texturecache'
        assert row['fingerprint'][5] == 1013
    return rows


def reclaim(manifest, confirmation, out):
    rows = verified_manifest(manifest)
    backup = json.loads(confirmation.read_bytes())
    assert backup['status'] == 'NEW_CACHE_RECOVERY_PRIVATE_D_ALL_FILES_VERIFIED'
    assert backup['archive']['sha256'] == ARCHIVE_SHA and backup['manifest']['sha256'] == MANIFEST_SHA
    for key in ['full_gzip_CRC_and_per_file_SHA_verified', 'current_native_Windows_owner_verified',
                'no_reparse', 'fsync_and_rename', 'transport_reaped', 'streams_final']:
        assert backup[key] is True
    assert backup['transport_exit_code'] == 0 and backup['verified_files'] == 549
    actual = set()
    for root in ROOTS:
        safe_parents(root)
        assert root.stat().st_uid == 1013
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = pathlib.Path(directory)
            actual.add(str(parent))
            for name in dirs + files:
                child = parent / name
                if child.is_symlink():
                    assert any(x['path'] == str(child) and x['kind'] == 'symlink' for x in rows)
                    actual.add(str(child))
            dirs[:] = [name for name in dirs if not (parent / name).is_symlink()]
            actual.update(str(parent / name) for name in files)
    assert actual == {x['path'] for x in rows}, 'Changed membership rejects the whole removal'
    for row in rows:
        p = pathlib.Path(row['path'])
        safe_parents(p.parent if row['kind'] == 'symlink' else p)
        s = p.lstat()
        assert fingerprint(s) == row['fingerprint'], str(p)
        if row['kind'] == 'symlink':
            assert stat.S_ISLNK(s.st_mode) and os.readlink(p) == row['target']
        if row['kind'] == 'file':
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and sha(p) == row['sha256']
    limits = idle()
    before = shutil.disk_usage(ROOTS[0]).free
    persist(out / 'intent.private.json', dict(backup_receipt_sha256=sha(confirmation),
        manifest_sha256=sha(manifest), source_entries_verified=len(rows), source_files_verified=549,
        data_free_before=before, readable_processes_no_references=True, service_visibility_limits=limits))
    removed = allocated = 0
    with (out / 'unlink_journal.private.jsonl').open('x') as journal:
        for row in rows:
            if row['kind'] != 'file':
                continue
            p = pathlib.Path(row['path'])
            safe_parents(p.parent)
            fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                s = os.stat(p.name, dir_fd=fd, follow_symlinks=False)
                assert fingerprint(s) == row['fingerprint'] and s.st_nlink == 1
                os.unlink(p.name, dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            removed += 1
            allocated += row['allocated_bytes']
            journal.write(json.dumps(dict(path=str(p), sha256=row['sha256'], removed=True)) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        for row in sorted(rows, key=lambda x: len(pathlib.Path(x['path']).parts), reverse=True):
            p = pathlib.Path(row['path'])
            if row['kind'] == 'directory' and p not in ROOTS:
                safe_parents(p)
                assert p.stat().st_uid == 1013
                if not list(p.iterdir()):
                    p.rmdir()
    assert removed == 549
    links = [x for x in rows if x['kind'] == 'symlink']
    for row in links:
        assert os.readlink(row['path']) == row['target']
    remaining_files = []
    for root in ROOTS:
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                q = pathlib.Path(directory) / name
                if not q.is_symlink():
                    remaining_files.append(str(q))
    assert remaining_files == []
    result = dict(status='NEW_IDLE_CACHES_PRIVATE_RECOVERY_VERIFIED_AND_RECLAIMED',
        removed_cache_files=removed, removed_file_allocated_bytes=allocated,
        original_tree_allocated_bytes=sum(x['allocated_bytes'] for x in rows),
        data_free_before=before, data_free_after=shutil.disk_usage(ROOTS[0]).free,
        private_archive_sha256=ARCHIVE_SHA, manifest_sha256=MANIFEST_SHA,
        data_checkpoint_environment_or_primary_source_files_removed=0,
        production_signals=0, new_model_or_GPU_jobs=0, roots_retained=True,
        known_external_cache_symlink_and_target_retained=True,
        global_fd_visibility_complete=not bool(limits), service_visibility_limit_count=len(limits))
    persist(out / 'receipt.json', result)
    return result


def restore(archive, manifest, out):
    rows = verified_manifest(manifest)
    assert sha(archive) == ARCHIVE_SHA
    lookup = {x['member'].rstrip('/.'): x for x in rows}
    for root in ROOTS:
        safe_parents(root)
        assert root.stat().st_uid == 1013
    count = 0
    with tarfile.open(archive, 'r:gz') as tar:
        seen = set()
        for member in tar:
            key = member.name.rstrip('/.')
            assert key in lookup and key not in seen
            seen.add(key)
            row = lookup[key]
            p = pathlib.Path(row['path'])
            safe_parents(p.parent if row['kind'] == 'symlink' else p)
            if row['kind'] == 'symlink':
                assert member.issym() and member.linkname == row['target']
                if p.is_symlink():
                    assert os.readlink(p) == row['target'] and p.lstat().st_uid == 1013
                else:
                    assert not p.exists()
                    p.symlink_to(row['target'])
                continue
            if row['kind'] == 'directory':
                assert member.isdir()
                p.mkdir(mode=row['fingerprint'][6], exist_ok=True)
                assert p.is_dir() and p.stat().st_uid == 1013
                continue
            assert member.isfile() and member.size == row['logical_bytes']
            if p.exists():
                assert p.is_file() and p.stat().st_uid == 1013 and sha(p) == row['sha256']
                continue
            partial = p.with_name(p.name + '.cache_recovery.partial')
            assert not partial.exists() and not partial.is_symlink()
            h = hashlib.sha256()
            with partial.open('xb') as stream:
                os.chmod(partial, row['fingerprint'][6])
                source = tar.extractfile(member)
                for block in iter(lambda: source.read(8388608), b''):
                    stream.write(block)
                    h.update(block)
                stream.flush()
                os.fsync(stream.fileno())
            assert h.hexdigest() == row['sha256'] and not p.exists()
            partial.rename(p)
            fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            count += 1
        assert len(seen) == len(rows)
    for row in sorted(rows, key=lambda x: len(pathlib.Path(x['path']).parts), reverse=True):
        p = pathlib.Path(row['path'])
        if row['kind'] == 'symlink':
            assert os.readlink(p) == row['target']
            continue
        os.chmod(p, row['fingerprint'][6])
        os.utime(p, ns=(row['fingerprint'][3], row['fingerprint'][3]))
        if row['kind'] == 'file':
            assert sha(p) == row['sha256']
    result = dict(status='EXACT_IDLE_CACHE_RECOVERY_FINISHED', restored_files=count,
                  verified_files=549, private_archive_sha256=ARCHIVE_SHA)
    persist(out / 'receipt.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['reclaim', 'restore'], required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--archive')
    p.add_argument('--private-d-receipt')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.getuid() == 1013
    os.sched_setaffinity(0, {0, 1, 2, 3})
    lock = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/idle_cache_cleanup_20261029.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = pathlib.Path(a.output)
    safe_parents(out)
    out.mkdir(mode=0o700, exist_ok=False)
    if a.mode == 'reclaim':
        result = reclaim(pathlib.Path(a.manifest), pathlib.Path(a.private_d_receipt), out)
    else:
        result = restore(pathlib.Path(a.archive), pathlib.Path(a.manifest), out)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
