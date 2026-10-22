"""One explicitly authorized 2026-10-22 archive-before-cleanup operation.

Uses existing local SSH/gh authentication. No credential goes to the server.
Only old native-contact JSON and the identified generated CLoSD output qualify.
Archives stream to local disk; no server staging space is required.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('backup_worker', BASE / 'scripts/backup_full_run.py')
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)
CONFIG = json.loads((BASE / 'configs/full_backup_worker_v1.json').read_text(encoding='utf-8'))
worker = worker_module.Worker(CONFIG)
LOCAL = Path((worker.root / 'current_storage_cleanup_local.txt').read_text(encoding='utf-8'))
RESULT = Path((worker.root / 'current_storage_cleanup_result.txt').read_text(encoding='utf-8'))
REMOTE = (LOCAL / 'remote_cleanup_root.txt').read_text(encoding='utf-8')
TAG = (worker.root / 'current_storage_cleanup_tag.txt').read_text(encoding='utf-8')
publisher = worker_module.Worker({**CONFIG, 'release_tag': TAG})
PRE = json.loads((worker.root / 'disk_cleanup_selected_preflight.json').read_text(encoding='utf-8'))
PREFIX = '/data/code/ouyanghao/oyh/conditional-human-motion-generation'
PIECE_SIZE = 1024 ** 3


def save(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path in [LOCAL / name, RESULT / name]:
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError('Refuse to replace prior archive evidence: ' + str(path))
        if not path.exists():
            path.write_bytes(data)


def log(event, **fields):
    row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event, **fields}
    with (LOCAL / 'archive_events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


class JoinedPieces:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None

    def read(self, n):
        if n < 0:
            raise ValueError('Unbounded archive reads are unsupported')
        chunks = []
        left = n
        while left:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file = path.open('rb')
            data = self.file.read(left)
            if not data:
                self.file.close()
                self.file = None
                continue
            chunks.append(data)
            left -= len(data)
        return b''.join(chunks)

    def close(self):
        if self.file is not None:
            self.file.close()


def inventory(group, paths):
    script = '''from pathlib import Path
import hashlib,json,os,stat,time
rows=[]
for name in NAMES:
 p=Path(name);s=p.lstat()
 if p.is_symlink() or p.resolve()!=p or not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid():raise ValueError('Invalid source path/owner')
 if time.time()-s.st_mtime<7*86400:raise ValueError('Recent result retained')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.stat()
 if (s.st_size,s.st_mtime_ns,s.st_ino)!=(z.st_size,z.st_mtime_ns,z.st_ino):raise ValueError('Source changed during hashing')
 rows.append({'path':str(p),'archive_relative_path':str(p.relative_to(Path(PREFIX).parent)),'bytes':s.st_size,'sha256':h.hexdigest(),'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'owner_uid':s.st_uid,'allocated_bytes':s.st_blocks*512})
print(json.dumps(rows))
'''
    script = 'NAMES=' + repr(paths) + '\nPREFIX=' + repr(PREFIX) + '\n' + script
    rows = json.loads(worker.remote(script, timeout=900))
    if group == 'closd_generated_20261010' and rows[0]['sha256'] != 'd2157b3a41681aa74472ec23beb0ec50843c6d1e10b66a4904ae05a49eafd05c':
        raise ValueError('Generated CLoSD output differs from its original generation receipt')
    save(group + '_source_manifest.json', {'group': group, 'files': rows})
    return rows


def stream_archive(group, rows):
    script = '''from pathlib import Path
import sys,tarfile,os,gzip
os.nice(10)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as gz, tarfile.open(fileobj=gz,mode='w|') as tar:
 for row in ROWS:
  p=Path(row['path']);s=p.lstat()
  if p.is_symlink() or p.resolve()!=p or (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['inode']):raise ValueError('Source changed before archive')
  tar.add(p,arcname=row['archive_relative_path'],recursive=False)
  s=p.lstat()
  if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['inode']):raise ValueError('Source changed during archive')
'''
    script = 'ROWS=' + repr(rows) + '\n' + script
    stderr_path = LOCAL / (group + '_stream_stderr.txt')
    parts = []
    total = 0
    full_hash = hashlib.sha256()
    output = None
    partial = None
    piece_bytes = 0
    last_notice = time.monotonic()
    with stderr_path.open('wb') as stderr:
        proc = subprocess.Popen(worker.ssh + [CONFIG['python'] + ' -'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=stderr)
        proc.stdin.write(script.encode('utf-8'))
        proc.stdin.close()
        try:
            while True:
                block = proc.stdout.read(4 * 1048576)
                if not block:
                    break
                full_hash.update(block)
                total += len(block)
                offset = 0
                while offset < len(block):
                    if output is None:
                        final = LOCAL / f'{group}.tar.gz.part{len(parts)+1:03d}'
                        partial = final.with_suffix(final.suffix + '.partial')
                        if final.exists() or partial.exists():
                            raise RuntimeError('Refuse to overwrite an earlier archive attempt')
                        output = partial.open('wb')
                        piece_bytes = 0
                    amount = min(PIECE_SIZE - piece_bytes, len(block) - offset)
                    output.write(block[offset:offset+amount])
                    offset += amount
                    piece_bytes += amount
                    if piece_bytes == PIECE_SIZE:
                        output.close()
                        output = None
                        parts.append((partial, final))
                if time.monotonic() - last_notice > 20:
                    log('archive_stream_progress', group=group, compressed_bytes=total)
                    last_notice = time.monotonic()
            if output is not None:
                output.close()
                output = None
                parts.append((partial, final))
            code = proc.wait(timeout=120)
            if code:
                raise RuntimeError(f'Archive stream exit {code}; see {stderr_path}')
        except BaseException:
            if output is not None:
                output.close()
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            raise
    for source, final in parts:
        os.replace(source, final)
    paths = [p[1] for p in parts]
    log('archive_stream_complete', group=group, bytes=total, pieces=len(paths))
    return paths, full_hash.hexdigest(), total


def verify_members(paths, rows):
    expected = {row['archive_relative_path']: row for row in rows}
    seen = set()
    joined = JoinedPieces(paths)
    try:
        with tarfile.open(fileobj=joined, mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected archive entry')
                row = expected[member.name]
                if member.size != row['bytes']:
                    raise ValueError('Archive size mismatch')
                h = hashlib.sha256()
                with tar.extractfile(member) as f:
                    for block in iter(lambda: f.read(1048576), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    raise ValueError('Restored member bytes differ from server inventory')
                seen.add(member.name)
    finally:
        joined.close()
    if seen != set(expected):
        raise ValueError('Missing archive members')


def run_guard(group, plan_path):
    worker.call([CONFIG['scp_executable'], '-o', 'BatchMode=yes', '-i', CONFIG['ssh_key'],
                 str(plan_path), CONFIG['ssh_host'] + ':' + REMOTE + '/' + plan_path.name], timeout=120)
    for execute in [False, True]:
        name = group + ('_cleanup_executed.json' if execute else '_dry_run.json')
        cmd = [CONFIG['python'], REMOTE + '/storage_cleanup_verified_v2.py', '--plan',
               REMOTE + '/' + plan_path.name, '--receipt', REMOTE + '/' + name]
        if execute:
            cmd.append('--execute')
        script = 'import subprocess,json\nr=subprocess.run(' + repr(cmd) + ',capture_output=True,text=True)\n'
        script += 'print(json.dumps({"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}))\n'
        result = json.loads(worker.remote(script, timeout=900))
        save(name + '.launcher.json', result)
        if result['returncode']:
            raise RuntimeError(result['stderr'])
        receipt = json.loads(result['stdout'])
        save(name, receipt)
        publisher.upload(LOCAL / name, publisher.assets())
        log('cleanup_guard_complete', group=group, mode=receipt['mode'],
            files=receipt['verified_files'], removed=len(receipt['removed']), bytes=receipt['logical_bytes'])


def archive_group(group, paths):
    log('inventory_started', group=group, files=len(paths))
    rows = inventory(group, paths)
    pieces, sha, size = stream_archive(group, rows)
    verify_members(pieces, rows)
    log('all_archive_members_byte_verified', group=group, files=len(rows))
    assets = publisher.assets()
    published = []
    for path in pieces:
        log('github_piece_upload_started', group=group, name=path.name, bytes=path.stat().st_size)
        record = publisher.upload(path, assets)
        published.append({**record, 'state': 'uploaded'})
        log('github_piece_digest_verified', group=group, name=path.name)
    index = {'group': group, 'files': rows, 'archive_sha256': sha, 'archive_bytes': size,
             'pieces_in_order': published, 'all_archive_member_bytes_sha256_verified_locally': True,
             'restore': 'Concatenate piece bytes in listed order, verify full archive SHA256, then restore only listed member paths and verify each source SHA256. Never load a pickle to verify backup bytes.'}
    save(group + '_backup_index.json', index)
    publisher.upload(LOCAL / (group + '_backup_index.json'), assets)
    plan = {'id': LOCAL.name, 'files': rows, 'backup_verification': {
        'assets': published, 'all_archive_member_bytes_sha256_verified_locally': True,
        'archive_member_count': len(rows), 'archive_members': [r['archive_relative_path'] for r in rows],
        'archive_sha256': sha, 'archive_bytes': size}}
    save(group + '_cleanup_plan.json', plan)
    plan_path = LOCAL / (group + '_cleanup_plan.json')
    publisher.upload(plan_path, assets)
    # Recheck every published chunk immediately before the destructive stage.
    fresh = publisher.assets()
    for asset in published:
        item = fresh[asset['name']]
        if item['state'] != 'uploaded' or item['size'] != asset['bytes'] or item.get('digest') != 'sha256:' + asset['sha256']:
            raise RuntimeError('Archive publication changed; no cleanup')
    run_guard(group, plan_path)


if __name__ == '__main__':
    raise SystemExit('This one-off cleanup is complete. Do not rerun it. The reusable CLoSD generated motion collection is protected and restored; see docs/data_preservation.md. Exact executed source remains in Git history at bc58298.')
