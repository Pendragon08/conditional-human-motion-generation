"""Download explicitly authorized official AMASS exports; credentials stay in memory."""
import argparse
import bz2
import getpass
import hashlib
import http.cookiejar
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_archive(path):
    records = []
    unpacked = 0
    with bz2.BZ2File(path, 'rb') as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                p = PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts or '\\' in member.name:
                    raise ValueError('Unsafe archive path')
                if not member.isfile() and not member.isdir():
                    raise ValueError('Links and special archive entries are unsupported')
                if member.isfile():
                    unpacked += member.size
                    records.append({'path': member.name, 'bytes': member.size})
        # Finish the compressed stream, including its CRC, beyond tar end markers.
        while stream.read(4 * 1024 * 1024):
            pass
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'archive_bytes': path.stat().st_size,
            'unpacked_file_bytes': unpacked, 'regular_files': len(records),
            'npz_files': sum(r['path'].endswith('.npz') for r in records),
            'members': records, 'compressed_crc_and_tar_headers_verified': True,
            'motion_arrays_validated': False, 'extracted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    assert queue['human_approved'] is True
    assert 'TotalCapture' not in [x['subset'] for x in queue['items']]
    for row in queue['items']:
        url = urllib.parse.urlparse(row['url'])
        assert url.scheme == 'https' and url.hostname == 'download.is.tue.mpg.de'
        assert row['license_approved'] is True
    args.destination.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=False)
    jar = http.cookiejar.CookieJar()  # This session is created here, never exported from a browser.
    session = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    session.addheaders = [('User-Agent', 'AMASS research asset acquisition')]
    first_url = queue['items'][0]['url']
    with session.open(first_url, timeout=60) as response:
        response.read(65536)
    username = getpass.getpass('Official AMASS account (hidden): ')
    password = getpass.getpass('Official AMASS password (hidden): ')
    form = urllib.parse.urlencode({'username': username, 'password': password, 'commit': 'Log in'}).encode()
    with session.open(urllib.request.Request(first_url, data=form), timeout=60) as response:
        binary = response.headers.get('Content-Disposition') is not None
        prefix = response.read(4)
        if not binary and prefix != b'BZh9' and not prefix.startswith(b'BZh'):
            raise RuntimeError('Official file-server login did not return an archive; no password retry performed')
    del username, password, form
    print('OFFICIAL_FILE_SERVER_LOGIN_ACCEPTED', flush=True)
    inventory = []
    for row in queue['items']:
        request = urllib.request.Request(row['url'], method='HEAD')
        with session.open(request, timeout=60) as response:
            kind = response.headers.get('Content-Type', '')
            if 'text/html' in kind:
                raise RuntimeError('Authenticated HEAD returned a login page')
            length = response.headers.get('Content-Length')
            inventory.append({**row, 'server_archive_bytes': int(length) if length else None,
                              'content_type': kind})
    save_json(args.receipts / 'remote_sizes.json', {'items': inventory,
              'total_known_archive_bytes': sum(x['server_archive_bytes'] or 0 for x in inventory),
              'all_sizes_known': all(x['server_archive_bytes'] is not None for x in inventory)})
    print(json.dumps({'REMOTE_SIZE_INVENTORY': [{k: x[k] for k in ('subset', 'server_archive_bytes')}
                                                for x in inventory]}, ensure_ascii=False), flush=True)
    # Complete smaller packages first, keeping a useful verified batch if storage becomes limiting.
    inventory.sort(key=lambda x: x['server_archive_bytes'] if x['server_archive_bytes'] is not None else 10**15)
    completed = []
    for row in inventory:
        subset = row['subset']
        final = args.destination / row['filename']
        partial = final.with_name(final.name + '.part')
        if final.exists() or partial.exists():
            raise FileExistsError('Refusing to overwrite prior acquisition: ' + str(final))
        expected = row['server_archive_bytes']
        if expected is not None and expected + 10 * 1024**3 > shutil.disk_usage(args.destination).free:
            raise RuntimeError('Insufficient space while retaining a 10 GiB reserve')
        print('DOWNLOAD_START ' + subset, flush=True)
        start = time.monotonic()
        last = start
        with session.open(row['url'], timeout=60) as response, partial.open('xb') as out:
            prefix = response.read(4)
            if not prefix.startswith(b'BZh'):
                raise RuntimeError('Download is not bzip2; refusing login/error HTML')
            out.write(prefix)
            downloaded = 4
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                if len(chunk) + 10 * 1024**3 > shutil.disk_usage(args.destination).free:
                    raise RuntimeError('Space reserve reached; partial retained')
                out.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last > 15:
                    print(json.dumps({'subset': subset, 'downloaded_bytes': downloaded,
                                      'expected_bytes': expected}), flush=True)
                    last = time.monotonic()
        if expected is not None and partial.stat().st_size != expected:
            raise RuntimeError('Size mismatch; partial retained')
        print('ARCHIVE_CRC_CHECK_START ' + subset, flush=True)
        verification = validate_archive(partial)
        partial.rename(final)
        receipt = {**row, **verification, 'status': 'LOCAL_ARCHIVE_CRC_AND_TAR_VERIFIED',
                   'elapsed_seconds': time.monotonic() - start,
                   'local_archive': str(final), 'server_deployed': False,
                   'body_model_weights_included': False, 'g1_pairs_generated': 0,
                   'new_262d_sequences': 0, 'generator_training_started': False}
        save_json(args.receipts / (subset + '.json'), receipt)
        completed.append({k: receipt[k] for k in ('subset', 'archive_bytes', 'unpacked_file_bytes', 'sha256')})
        save_json(args.receipts / 'completed.json', {'completed': completed,
                  'full_amass_downloaded': False, 'totalcapture_skipped_by_user': True})
        print('ARCHIVE_VERIFIED ' + json.dumps(completed[-1]), flush=True)
    print('AUTHORIZED_BATCH_DOWNLOADED_AND_ARCHIVE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
