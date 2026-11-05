"""Upload pre-validated GRAB archives without overwriting; resume verified prefixes."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(r'D:\具身智能\DATASET')
REMOTE = '/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/GRAB'
HOST = '10.147.17.32'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', HOST]
LOG = pathlib.Path(r'D:\GRAB\grab_checked_upload_20261030.jsonl')
LOCK = REMOTE + '/.checked_upload_lock'


def record(event, **fields):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': time.time(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def remote(command, check=True):
    return subprocess.run(SSH + [command], capture_output=True, text=True, timeout=120, check=check)


def sha(path, limit=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while limit is None or limit:
            b = f.read(8 * 1024 * 1024 if limit is None else min(limit, 8 * 1024 * 1024))
            if not b:
                break
            h.update(b)
            if limit is not None:
                limit -= len(b)
    if limit not in (None, 0):
        raise ValueError('local prefix shorter than expected')
    return h.hexdigest()


def main():
    report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf8'))
    if remote('mkdir ' + LOCK, check=False).returncode != 0:
        raise RuntimeError('upload lock already held or server unavailable')
    try:
        for row in report:
            name = row['name']
            if not re.fullmatch(r'(?:grab__s(?:1|3|4|5|6|7|8|9|10)|tools__subject_meshes__(?:male|female)|tools__object_meshes__contact_meshes)\.zip', name):
                raise ValueError('unexpected filename: ' + name)
            if row['bad'] is not None or not row['stable']:
                raise ValueError('archive failed integrity verification')
            path = ROOT / name
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('local source differs from validated archive: ' + name)
            final = REMOTE + '/' + name
            partial = final + '.part'
            result = remote(f'if test -e {final}; then sha256sum {final}; else echo ABSENT; fi').stdout.strip()
            if result != 'ABSENT':
                if result.split()[0] != row['sha256']:
                    raise ValueError('server destination conflict: ' + name)
                record('already_verified', name=name, sha256=row['sha256'])
                continue
            result = remote(f'if test -e {partial}; then stat -c %s {partial}; else echo ABSENT; fi').stdout.strip()
            if result == 'ABSENT':
                remote(f'set -C; : > {partial}')
                n = 0
            else:
                n = int(result)
                if n > row['bytes']:
                    raise ValueError('partial larger than local source: ' + name)
            if n:
                remote_hash = remote(f'head -c {n} {partial} | sha256sum').stdout.split()[0]
                if remote_hash != sha(path, n):
                    raise ValueError('partial prefix differs from local file: ' + name)
            record('transfer_start', name=name, resumed_bytes=n)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', suffix='.sftp', delete=False) as f:
                batch = pathlib.Path(f.name)
                f.write(f'reput "{path.as_posix()}" "{partial}"\n')
            try:
                proc = subprocess.run(['sftp', '-B', '65536', '-R', '64', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', '-b', str(batch), HOST], capture_output=True, text=True, timeout=1800)
            finally:
                batch.unlink(missing_ok=True)
            if proc.returncode != 0:
                raise RuntimeError(f'sftp failed for {name}: {proc.stderr[-500:]}')
            remote_hash = remote(f'sha256sum {partial}').stdout.split()[0]
            if remote_hash != row['sha256'] or sha(path) != row['sha256']:
                raise ValueError('full-file hash mismatch: ' + name)
            remote(f'ln {partial} {final} && rm {partial}')
            if remote(f'sha256sum {final}').stdout.split()[0] != row['sha256']:
                raise ValueError('published hash mismatch: ' + name)
            record('verified', name=name, bytes=row['bytes'], sha256=row['sha256'])
        record('complete', count=len(report))
    finally:
        remote('rmdir ' + LOCK, check=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('failed', error=str(exc))
        raise
