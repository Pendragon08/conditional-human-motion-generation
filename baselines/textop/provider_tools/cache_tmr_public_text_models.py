"""Pinned public HF cache via the existing local proxy, preserving all partials.

This is an alternative transport after the server's direct route genuinely
failed. It never launches SSH, trains or reads robot/BABEL payloads.
"""
import hashlib
import json
import msvcrt
import os
import pathlib
import shutil
import threading
import time
import traceback
import urllib.error
import urllib.request
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
O = B / 'results/semantic_evaluator_20261029/public_text_cache_v1'
O.mkdir(exist_ok=False)
P = D / 'models/evaluation_text_tmr_20261029'
guard_local(D, P, directory=True, must_absent=True)
P.mkdir(parents=True, mode=0o700)
assert shutil.disk_usage(D).free > 2 * 1024**3 + 707323471 + 128 * 1024**2
prior = json.loads((B / 'results/semantic_evaluator_20261029/tmr_text_models_v1/progress.json').read_bytes())
assert prior['files']['outer']['actual_returncode'] == 1 and not prior['workers'] and prior['common_lock_free']
assert 'Network is unreachable' in prior['files']['receipt']['error'] and not prior['files']['receipt']['completed_files']
m = json.loads((B / 'results/semantic_evaluator_20261029/public_dependency_manifest_v2.json').read_bytes())
lock = (P / 'public_cache.lock').open('a+b')
lock.seek(0)
msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
start = time.monotonic()
stop = threading.Event()
sampler_errors = []
receipt = dict(status='DOWNLOADING_PINNED_PUBLIC_TMR_TEXT_FILES', run='public_text_cache_v1', pid=os.getpid(),
    completed_files=[], output_writers_closed=False, model_payload_loads=0, optimizer_updates=0,
    private_robot_or_text_reads=0, original_server_failure_retained=True, proxy='http://127.0.0.1:7897')


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save():
    receipt['elapsed_seconds'] = time.monotonic() - start
    tmp = O / 'receipt.partial'
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(O / 'receipt.json')


def sampler():
    try:
        with (O / 'resources.jsonl').open('x', encoding='utf-8') as f:
            while True:
                free = shutil.disk_usage(D).free
                f.write(json.dumps(dict(unix=time.time(), D_free_bytes=free, pid=os.getpid(),
                                        elapsed_seconds=time.monotonic() - start)) + '\n')
                f.flush()
                assert free > 2 * 1024**3 and time.monotonic() - start < 3600
                if stop.wait(15):
                    break
    except BaseException as e:
        sampler_errors.append(repr(e))


write(O / 'intent.json', dict(source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    prior_closed_failure_receipt_sha256=prior['sha256']['receipt'], exact_model_file_bytes=707323471,
    no_server_redispatch=True, deadline_seconds=3600, D_initial_free_bytes=shutil.disk_usage(D).free))
save()
thread = threading.Thread(target=sampler, daemon=True)
thread.start()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': receipt['proxy'], 'https': receipt['proxy']}))
exit_code = 1
try:
    for mi, model in enumerate(m['models']):
        folder = P / ('text_model_' + str(mi))
        folder.mkdir()
        for item in model['files']:
            dest = folder / item['name']
            guard_local(D, dest, must_absent=True)
            part = dest.with_name(dest.name + '.partial')
            write(folder / (item['name'] + '.intent.json'), item)
            with part.open('xb'):
                pass
            for attempt in range(3):
                offset = part.stat().st_size
                attempt_record = dict(offset=offset, prefix_sha256=digest(part), attempt=attempt)
                try:
                    assert not sampler_errors and time.monotonic() - start < 3600
                    if offset < item['bytes']:
                        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'TextOp-pinned-public-evaluator/1.0'}
                        if offset:
                            headers['Range'] = 'bytes={}-'.format(offset)
                        with opener.open(urllib.request.Request(item['url'], headers=headers), timeout=90) as response:
                            if offset:
                                assert response.status == 206 and response.headers['Content-Range'].startswith('bytes {}-'.format(offset))
                                assert response.headers['Content-Range'].endswith('/' + str(item['bytes']))
                            else:
                                assert response.status == 200
                            with part.open('ab') as f:
                                while True:
                                    block = response.read(1024**2)
                                    if not block:
                                        break
                                    assert not sampler_errors and time.monotonic() - start < 3600
                                    assert shutil.disk_usage(D).free > 2 * 1024**3 + 128 * 1024**2 + len(block)
                                    assert f.tell() + len(block) <= item['bytes']
                                    f.write(block)
                                f.flush()
                                os.fsync(f.fileno())
                    assert part.stat().st_size == item['bytes']
                    if item['sha256']:
                        assert digest(part) == item['sha256']
                    else:
                        raw = part.read_bytes()
                        assert hashlib.sha1(('blob %d\0' % len(raw)).encode() + raw).hexdigest() == item['git_blob_sha']
                    owner = guard_local(D, part)
                    part.replace(dest)
                    assert digest(dest) == (item['sha256'] or digest(dest))
                    attempt_record.update(completed=True, sha256=digest(dest), bytes=dest.stat().st_size, native_owner=owner)
                    write(folder / (item['name'] + '.attempt' + str(attempt) + '.json'), attempt_record)
                    receipt['completed_files'].append(dict(repository=model['repository'], revision=model['revision'],
                        path=str(dest), bytes=dest.stat().st_size, sha256=digest(dest), native_owner_verified=True))
                    save()
                    break
                except (OSError, TimeoutError, urllib.error.URLError) as e:
                    attempt_record.update(completed=False, error=repr(e), partial_bytes=part.stat().st_size)
                    write(folder / (item['name'] + '.attempt' + str(attempt) + '.json'), attempt_record)
                    if attempt == 2:
                        raise
            assert dest.exists()
    assert sum(x['bytes'] for x in receipt['completed_files']) == 707323471
    receipt.update(status='PUBLIC_TMR_TEXT_CACHE_D_BYTES_VERIFIED', output_writers_closed=True,
                   native_owner=guard_local(D, P, directory=True), model_payload_loads=0, optimizer_updates=0)
    exit_code = 0
except BaseException as e:
    receipt.update(status='FAILED_PUBLIC_TMR_TEXT_CACHE_ALL_PARTIALS_RETAINED', error=repr(e), traceback=traceback.format_exc())
finally:
    stop.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    receipt['resource_sampler_errors'] = sampler_errors
    save()
    write(O / 'exit.json', dict(actual_returncode=exit_code, resource_sampler_joined=True, sampler_errors=sampler_errors))
    print(json.dumps(dict(status=receipt['status'], completed_files=len(receipt['completed_files']), actual_returncode=exit_code)), flush=True)
raise SystemExit(exit_code)
