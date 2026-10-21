"""Resumable, bounded asset batch. Detached workers receive credentials through a pipe."""
import argparse
import copy
import datetime
import getpass
import hashlib
import http.cookiejar
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from download_amass_archives import save_json, validate_archive
from stage_licensed_archive import stage


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--receipts', type=Path, required=True)
    p.add_argument('--previous-receipts', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--remote-python', required=True)
    p.add_argument('--credential-stdin', action='store_true')
    a = p.parse_args()
    if not a.credential_stdin:
        username = getpass.getpass('Official asset account (hidden): ')
        password = getpass.getpass('Official asset password (hidden): ')
        a.receipts.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--credential-stdin']
        with (a.receipts / 'worker.stdout.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
            child.stdin.write(json.dumps({'username': username, 'password': password}).encode())
            child.stdin.close()
        del username, password
        save_json(a.receipts / 'launch.json', {'pid': child.pid, 'credentials_stored': False,
                  'credential_transport': 'anonymous stdin pipe only', 'queue_sha256': sha(a.queue),
                  'worker_sha256': sha(Path(__file__)),
                  'archive_validator_sha256': sha(Path(__file__).with_name('download_amass_archives.py')),
                  'server_stager_sha256': sha(Path(__file__).with_name('stage_licensed_archive.py'))})
        print(json.dumps({'status': 'DETACHED_BATCH_STARTED', 'pid': child.pid, 'receipts': str(a.receipts)}))
        return

    account = json.load(sys.stdin)
    queue = json.loads(a.queue.read_text())
    assert queue['human_approved'] and all(x['license_approved'] for x in queue['items'])
    assert not any(x['subset'] == 'TotalCapture' for x in queue['items'])
    assert len({x['subset'] for x in queue['items']}) == len(queue['items'])
    a.destination.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    statuses = {}
    inventory = []
    jars = {}

    def event(subset, state, **fields):
        with lock:
            record = {'subset': subset, 'state': state, **fields}
            statuses[subset] = record
            save_json(a.receipts / 'progress.json', {
                'updated_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                'pid': os.getpid(), 'items': list(statuses.values()), 'full_amass_downloaded': False,
                'totalcapture_skipped_by_user': True})
            print(json.dumps(record), flush=True)

    def opener(jar):
        session = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        session.addheaders = [('User-Agent', 'AMASS research asset acquisition')]
        return session

    for row in queue['items']:
        parsed = urllib.parse.urlparse(row['url'])
        assert parsed.scheme == 'https' and parsed.hostname == 'download.is.tue.mpg.de'
        domain = urllib.parse.parse_qs(parsed.query)['domain'][0]
        assert domain in ('amass', 'smplx')
        previous = a.previous_receipts / (row['subset'] + '.json')
        previous_record = json.loads(previous.read_text()) if previous.exists() else None
        completed_file = a.destination / row['filename']
        if previous_record and completed_file.exists() and completed_file.stat().st_size == previous_record['archive_bytes']:
            # Existing licensed assets need no new login or network access; hash is checked below.
            inventory.append({**row, 'server_archive_bytes': previous_record['archive_bytes'], 'domain': domain})
            continue
        if domain not in jars:
            jar = http.cookiejar.CookieJar()
            client = opener(jar)
            with client.open(row['url'], timeout=90) as response:
                response.read(4096)
            form = urllib.parse.urlencode({**account, 'commit': 'Log in'}).encode()
            with client.open(urllib.request.Request(row['url'], data=form), timeout=90) as response:
                prefix = response.read(4)
                assert prefix.startswith((b'BZh', b'PK')), 'Official login did not authorize file access'
            jars[domain] = jar
            del form
        client = opener(jars[domain])
        with client.open(urllib.request.Request(row['url'], method='HEAD'), timeout=90) as response:
            assert 'text/html' not in response.headers.get('Content-Type', '')
            size = response.headers.get('Content-Length')
            assert size is not None, 'A known size is required for this disk-limited batch'
            inventory.append({**row, 'server_archive_bytes': int(size), 'domain': domain})
    save_json(a.receipts / 'remote_sizes.json', {'items': inventory,
              'total_archive_bytes': sum(x['server_archive_bytes'] for x in inventory)})
    stop_keepalive = threading.Event()

    def keepalive():
        while not stop_keepalive.wait(120):
            for domain, jar in jars.items():
                url = next(r['url'] for r in inventory if r['domain'] == domain)
                try:
                    client = opener(jar)
                    with client.open(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
                        expired = 'text/html' in response.headers.get('Content-Type', '')
                    if expired:
                        form = urllib.parse.urlencode({**account, 'commit': 'Log in'}).encode()
                        with client.open(urllib.request.Request(url, data=form), timeout=60) as response:
                            assert response.read(4).startswith((b'BZh', b'PK'))
                        del form
                    print('AUTHORIZED_SESSION_KEPT_ACTIVE ' + domain, flush=True)
                except Exception as exc:
                    print('SESSION_KEEPALIVE_ATTENTION ' + domain + ' ' + type(exc).__name__, flush=True)

    threading.Thread(target=keepalive, daemon=True).start()
    # Models first, then smaller motion packages; only three transfers at once.
    inventory.sort(key=lambda x: (x.get('kind') != 'model_zip', x['server_archive_bytes']))

    def acquire(row):
        subset = row['subset']
        final = a.destination / row['filename']
        partial = final.with_name(final.name + '.part')
        expected = row['server_archive_bytes']
        jar = http.cookiejar.CookieJar()
        for cookie in jars.get(row['domain'], []):
            jar.set_cookie(copy.copy(cookie))
        client = opener(jar)
        try:
            previous = a.previous_receipts / (subset + '.json')
            if final.exists():
                old = json.loads(previous.read_text())
                assert old['archive_bytes'] == expected == final.stat().st_size and old['sha256'] == sha(final)
                verification = {k: old[k] for k in ('sha256', 'archive_bytes', 'unpacked_file_bytes',
                                'regular_files', 'npz_files', 'members', 'compressed_crc_and_tar_headers_verified',
                                'zip_crc_verified') if k in old}
                event(subset, 'REUSED_PRIOR_VERIFIED_LOCAL_ARCHIVE', archive_bytes=expected)
            else:
                for attempt in range(8):
                    offset = partial.stat().st_size if partial.exists() else 0
                    assert offset <= expected
                    if offset == expected:
                        break
                    if expected - offset + 10 * 1024**3 > shutil.disk_usage(a.destination).free:
                        raise RuntimeError('D drive 10 GiB reserve would be violated')
                    headers = {'Range': 'bytes=' + str(offset) + '-'} if offset else {}
                    try:
                        with client.open(urllib.request.Request(row['url'], headers=headers), timeout=90) as response:
                            if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes ' + str(offset) + '-')):
                                raise RuntimeError('Server did not honor exact resume range; partial retained')
                            if 'text/html' in response.headers.get('Content-Type', ''):
                                raise RuntimeError('Login/error HTML refused')
                            last = 0
                            with partial.open('ab' if offset else 'xb') as out:
                                while True:
                                    chunk = response.read(4194304)
                                    if not chunk:
                                        break
                                    if offset == 0:
                                        assert chunk.startswith(b'PK' if row.get('kind') == 'model_zip' else b'BZh')
                                    out.write(chunk)
                                    offset += len(chunk)
                                    if time.monotonic() - last > 20:
                                        out.flush()
                                        event(subset, 'DOWNLOADING', downloaded_bytes=offset, expected_bytes=expected)
                                        last = time.monotonic()
                        if partial.stat().st_size != expected:
                            raise OSError('Incomplete transfer; verified resume offset retained')
                        break
                    except (OSError, urllib.error.URLError) as exc:
                        if attempt == 7:
                            raise
                        event(subset, 'RETRYING_RESUMABLE_NETWORK_ERROR', attempt=attempt + 1,
                              downloaded_bytes=partial.stat().st_size if partial.exists() else 0,
                              expected_bytes=expected, error_type=type(exc).__name__)
                        time.sleep(min(45, 3 * 2 ** attempt))
                assert partial.stat().st_size == expected
                event(subset, 'VERIFYING_LOCAL_ARCHIVE', archive_bytes=expected)
                if row.get('kind') == 'model_zip':
                    records = []
                    with zipfile.ZipFile(partial) as z:
                        assert z.testzip() is None
                        for info in z.infolist():
                            p = PurePosixPath(info.filename)
                            assert not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename
                            assert (info.external_attr >> 16) & 0o170000 != 0o120000
                            if not info.is_dir():
                                records.append({'path': info.filename, 'bytes': info.file_size})
                    verification = {'archive_bytes': expected, 'sha256': sha(partial), 'members': records,
                                    'regular_files': len(records), 'npz_files': sum(x['path'].endswith('.npz') for x in records),
                                    'unpacked_file_bytes': sum(x['bytes'] for x in records), 'zip_crc_verified': True}
                else:
                    verification = validate_archive(partial)
                partial.rename(final)
            receipt = {**row, **verification, 'status': 'LOCAL_ARCHIVE_VERIFIED', 'local_archive': str(final),
                       'g1_pairs_generated': 0, 'new_262d_sequences': 0, 'generator_training_started': False}
            save_json(a.receipts / (subset + '.json'), receipt)
            event(subset, 'LOCAL_ARCHIVE_VERIFIED', archive_bytes=expected, sha256=verification['sha256'])
            return {'subset': subset, 'local_archive_verified': True, 'receipt': receipt}
        except Exception as exc:
            event(subset, 'NEEDS_ATTENTION', error=type(exc).__name__ + ': ' + str(exc))
            return {'subset': subset, 'error': type(exc).__name__ + ': ' + str(exc), 'local_archive_verified': final.exists()}

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(acquire, row) for row in inventory]
        for future in as_completed(futures):
            result = future.result()
            receipt = result.pop('receipt', None)
            if receipt:
                subset = receipt['subset']
                try:
                    known = receipt.get('previous_server_receipt')
                    if known:
                        server_result = json.loads(Path(known).read_text())
                        assert server_result['archive_sha256'] == receipt['sha256']
                        target = a.receipts / 'server' / (subset + '.json')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        save_json(target, server_result)
                        server_result = {k:v for k,v in server_result.items() if k != 'files'}
                    else:
                        event(subset, 'SERVER_CAPACITY_CHECK', archive_bytes=receipt['archive_bytes'])
                        server_result = stage(Path(receipt['local_archive']), receipt, host=a.host, identity=a.identity,
                            remote_root=a.remote_root, remote_python=a.remote_python,
                            output=a.receipts / 'server' / (subset + '.json'))
                    save_json(a.receipts / (subset + '_server_status.json'), server_result)
                    result['server_status'] = server_result['status']
                    event(subset, server_result['status'], archive_bytes=receipt['archive_bytes'],
                          server_directory=server_result.get('server_directory'),
                          required_free_bytes=server_result.get('required_free_bytes_including_5gib_reserve'))
                except Exception as exc:
                    result['server_error'] = type(exc).__name__ + ': ' + str(exc)
                    event(subset, 'LOCAL_VERIFIED_SERVER_NEEDS_ATTENTION', error=result['server_error'])
            results.append(result)
            save_json(a.receipts / 'summary.json', {'results': results, 'full_amass_downloaded': False,
                      'totalcapture_skipped_by_user': True, 'running': len(results) != len(inventory)})
    stop_keepalive.set()
    account.clear()
    print('AUTHORIZED_BATCH_FINISHED ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
