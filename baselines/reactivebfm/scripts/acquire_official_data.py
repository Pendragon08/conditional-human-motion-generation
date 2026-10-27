"""Acquire the public author-provided files; retain partial transfers and hash all bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import threading
import urllib.parse
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''): h.update(b)
    return h.hexdigest()


def main(a):
    a.receipts.mkdir(parents=True, exist_ok=True)
    rows = json.loads(a.manifest.read_text())['files']
    rows.sort(key=lambda r: (r['dataset']!='kungfu_g1_36dim', r['bytes']))
    lock = threading.Lock(); records = []
    def save(row):
        with lock:
            records.append(row)
            (a.receipts/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    def acquire(row):
        path = a.destination/row['dataset']/row['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.part')
        try:
            if not path.exists():
                start = partial.stat().st_size if partial.exists() else 0
                urls = [row['url'], 'https://drive.google.com/uc?export=download&id='+row['id']]
                response = None
                for url in urls:
                    try:
                        request = urllib.request.Request(url, headers={'User-Agent':'Research reproduction', **({'Range':f'bytes={start}-'} if start else {})})
                        response = urllib.request.urlopen(request, timeout=120)
                        if 'text/html' in response.headers.get('Content-Type',''):
                            page = response.read().decode(); response.close()
                            form = re.search(r'<form[^>]+action="([^"]+)"', page)
                            if not form: raise RuntimeError('Official file returned HTML without a download form')
                            action = html.unescape(form.group(1))
                            if urllib.parse.urlparse(action).hostname not in ['drive.usercontent.google.com','drive.google.com']:
                                raise RuntimeError('Unexpected download form host')
                            params = {html.unescape(n):html.unescape(v) for n,v in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',page)}
                            response = urllib.request.urlopen(action+'?'+urllib.parse.urlencode(params),timeout=120)
                        if 'text/html' in response.headers.get('Content-Type',''): raise RuntimeError('File is unavailable; no data written')
                        if start and response.status != 206: raise RuntimeError('Resume unsupported; partial retained')
                        break
                    except Exception:
                        response = None
                if response is None: raise RuntimeError('Both official download endpoints failed')
                with response, partial.open('ab' if start else 'xb') as output:
                    while True:
                        block = response.read(1048576)
                        if not block: break
                        output.write(block)
                if partial.stat().st_size != row['bytes']: raise ValueError('Expected size differs; partial retained')
                partial.rename(path)
            if path.stat().st_size != row['bytes']: raise ValueError('Existing file size differs')
            result = {**row,'status':'SIZE_AND_SHA256_RECORDED','sha256':digest(path), 'local_path':str(path)}
        except Exception as exc:
            result = {**row,'status':'FAILED','error':type(exc).__name__+': '+str(exc),'partial_retained':partial.exists()}
        save(result)
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(acquire, rows))
    result = {'completed_at':datetime.datetime.now().astimezone().isoformat(),
        'files_requested':len(rows),'files_verified':sum(r['status']=='SIZE_AND_SHA256_RECORDED' for r in records),
        'bytes_verified':sum(r['bytes'] for r in records if r['status']=='SIZE_AND_SHA256_RECORDED'),
        'source':'Official ReactiveBFM Google Drive release','files':records,'no_data_deleted':True}
    (a.receipts/'acquisition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('manifest','destination','receipts'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
