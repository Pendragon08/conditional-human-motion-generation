"""Resumable read-only checkpoint backup over unstable SSH; never deletes files.

Every immutable chunk has a server SHA256 and byte count. The joined model and
optimizer must match the ready manifest before upload; GitHub's digests must
then match. Historical failures/partials are retained in separate directories.
No checkpoint is deserialized and no training/environment files are modified.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import threading
import time


SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=2', '-i', 'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32']
RUN = '/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/training_fp64_sdpa_v4'
REPO = 'Pendragon08/conditional-human-motion-generation'
TAG = 'paper-reproduction-progress-20261023'
GH = 'C:/Program Files/GitHub CLI/gh.exe'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(a):
    a.output.mkdir(parents=True, exist_ok=a.resume)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if manifest['optimizer_updates'] != a.expected_updates or len(manifest['files']) != 2:
        raise ValueError('Expected one exact model/optimizer checkpoint pair')
    expected_names = {f'model{manifest["official_step"]:09d}.pt', f'opt{manifest["official_step"]:09d}.pt'}
    if {PurePosixPath(r['file']).name for r in manifest['files']} != expected_names:
        raise ValueError('Wrong checkpoint filenames')
    if any(PurePosixPath(r['file']).parent != PurePosixPath(RUN)/'checkpoints' for r in manifest['files']):
        raise ValueError('Only the pinned checkpoint directory is allowed')
    lock = threading.Lock()
    state = {'status': 'COPYING_VERIFIED_CHUNKS', 'expected_updates': a.expected_updates,
        'deletes_files': False, 'gpu_used': False, 'manifest': manifest,
        'script_sha256': digest(__file__), 'chunk_bytes': a.chunk_mib*2**20,
        'parallel_connections': a.workers, 'completed_chunks': [], 'transfer_failures': []}
    if a.resume:
        previous = json.loads((a.output/'receipt.json').read_text(encoding='utf-8'))
        if previous['manifest'] != manifest or previous['chunk_bytes'] != state['chunk_bytes']:
            raise ValueError('Resume requires the identical immutable manifest/chunk layout')
        history = a.output/f'prior_receipt_{time.time_ns()}.json'
        history.write_text(json.dumps(previous, indent=2)+'\n', encoding='utf-8')
        state['completed_chunks'] = previous['completed_chunks']
        state['transfer_failures'] = previous['transfer_failures']
        state['resumed_from'] = str(history)
    prior = {(r['file'], r['offset']): r for r in state['completed_chunks']}

    def save():
        p = a.output/'receipt.json'
        tmp = a.output/'receipt.partial.json'
        tmp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        tmp.replace(p)

    def get_chunk(row, offset, count):
        name = PurePosixPath(row['file']).name
        directory = a.output/'verified_chunks'/name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory/f'{offset:012d}.bin'
        if (name, offset) in prior:
            record = prior[name, offset]
            if record['bytes'] != count or target.stat().st_size != count or digest(target) != record['sha256']:
                raise ValueError('Previously verified chunk changed; preserving it and refusing resume')
            return record
        code = '''from pathlib import Path
import hashlib,json,sys
p=Path(FILE)
assert p.parent==Path(PARENT) and p.is_file() and not p.is_symlink()
assert p.stat().st_size==TOTAL
with p.open('rb') as f:
 f.seek(OFFSET);b=f.read(COUNT)
assert len(b)==COUNT
sys.stdout.buffer.write((json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})+'\\n').encode())
sys.stdout.buffer.write(b)
'''
        for key, val in {'FILE': repr(row['file']), 'PARENT': repr(RUN+'/checkpoints'),
                         'TOTAL': str(row['bytes']), 'OFFSET': str(offset), 'COUNT': str(count)}.items():
            code = code.replace(key, val)
        for attempt in range(1, 6):
            started = time.monotonic()
            try:
                p = subprocess.run(SSH+['python3 -c '+shlex.quote(code)], capture_output=True, timeout=120)
                if p.returncode:
                    raise RuntimeError(p.stderr.decode(errors='replace'))
                header, payload = p.stdout.split(b'\n', 1)
                receipt = json.loads(header)
                if receipt['bytes'] != count or len(payload) != count or hashlib.sha256(payload).hexdigest() != receipt['sha256']:
                    raise ValueError('Chunk bytes/SHA256 mismatch')
                if target.exists():
                    if target.stat().st_size != count or digest(target) != receipt['sha256']:
                        raise ValueError('Uncommitted chunk differs from the server; preserving it')
                else:
                    with target.open('xb') as f:
                        f.write(payload)
                record = {'file': name, 'offset': offset, 'bytes': count,
                    'sha256': receipt['sha256'], 'elapsed_s': time.monotonic()-started}
                with lock:
                    state['completed_chunks'].append(record);save()
                    print(json.dumps({'completed_chunks': len(state['completed_chunks']),
                        'verified_bytes': sum(r['bytes'] for r in state['completed_chunks'])}), flush=True)
                return record
            except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                failure = {'file': name, 'offset': offset, 'bytes': count,
                    'attempt': attempt, 'error': repr(exc), 'elapsed_s': time.monotonic()-started}
                if isinstance(exc, subprocess.TimeoutExpired) and exc.stdout:
                    partial = directory/f'{offset:012d}.failed_{time.time_ns()}.bin'
                    partial.write_bytes(exc.stdout);failure['preserved_partial'] = str(partial)
                with lock:
                    state['transfer_failures'].append(failure);save()
                if attempt == 5:
                    raise
                time.sleep(3)

    try:
        save()
        tasks = [(r, start, min(state['chunk_bytes'], r['bytes']-start))
            for r in manifest['files'] for start in range(0, r['bytes'], state['chunk_bytes'])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(get_chunk, *task) for task in tasks]
            for f in concurrent.futures.as_completed(futures):
                f.result()
        copied = []
        for row in manifest['files']:
            name = PurePosixPath(row['file']).name
            target = a.output/('reactive_v4_'+name)
            if not target.exists():
                with target.open('xb') as output:
                    for offset in range(0, row['bytes'], state['chunk_bytes']):
                        chunk = a.output/'verified_chunks'/name/f'{offset:012d}.bin'
                        output.write(chunk.read_bytes())
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                raise ValueError('Joined checkpoint failed the original ready-manifest digest')
            copied.append({'name': target.name, 'local_path': str(target), 'bytes': row['bytes'], 'sha256': row['sha256']})
        state.update(status='LOCAL_MODEL_AND_OPTIMIZER_VERIFIED', copied=copied);save()
        for row in copied:
            info = json.loads(subprocess.check_output([GH, 'api', f'repos/{REPO}/releases/tags/{TAG}'], text=True, encoding='utf-8'))
            matches = [r for r in info['assets'] if r['name'] == row['name']]
            if matches:
                if len(matches) != 1 or matches[0]['size'] != row['bytes'] or matches[0]['digest'] != 'sha256:'+row['sha256']:
                    raise ValueError('Existing GitHub checkpoint differs; refusing overwrite')
            else:
                subprocess.run([GH, 'release', 'upload', TAG, row['local_path'], '--repo', REPO], check=True, timeout=3600)
        info = json.loads(subprocess.check_output([GH, 'api', f'repos/{REPO}/releases/tags/{TAG}'], text=True, encoding='utf-8'))
        assets = {r['name']:r for r in info['assets']}
        for row in copied:
            asset = assets[row['name']]
            if asset['size'] != row['bytes'] or asset['digest'] != 'sha256:'+row['sha256']:
                raise ValueError('GitHub size/digest mismatch')
            row['url'] = asset['browser_download_url']
        state.update(status='GITHUB_AND_LOCAL_VERIFIED', server_checkpoints_preserved=True);save()
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc));save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-updates', type=int, required=True)
    p.add_argument('--chunk-mib', type=int, default=16, choices=[4, 8, 16, 32])
    p.add_argument('--workers', type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument('--resume', action='store_true', help='Only resume the identical manifest; keep all earlier receipts and chunks')
    main(p.parse_args())
