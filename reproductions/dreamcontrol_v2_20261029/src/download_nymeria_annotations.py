"""Download the authorized Nymeria text/metadata groups with SHA-1 checks.

This intentionally excludes motion archives and videos. URL JSON contains
time-limited signed links: keep it private and never write it to logs. Existing
files are verified but never overwritten when their hashes differ.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import urllib.request
from urllib.error import HTTPError, URLError

OUTPUTS = {
    'metadata_json': 'metadata.json',
    'narration_motion_narration_csv': 'narration/motion_narration.csv',
    'narration_atomic_action_csv': 'narration/atomic_action.csv',
    'narration_activity_summarization_csv': 'narration/activity_summarization.csv',
}
SAFE_SEQUENCE = re.compile(r'^[A-Za-z0-9_-]+$')


def digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(seq: str, group: str, info: dict, output: Path) -> dict:
    path = output / seq / OUTPUTS[group]
    expected = info['sha1sum'].lower()
    size = int(info['file_size_bytes'])
    result = {'sequence': seq, 'group': group, 'bytes': size}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        result['status'] = 'verified_existing' if path.stat().st_size == size and digest(path) == expected else 'existing_mismatch'
        return result
    for attempt in range(3):
        partial = None
        try:
            req = urllib.request.Request(info['download_url'], headers={'User-Agent': 'DreamControl-v2 research data download'})
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f'HTTP {response.status}')
                h = hashlib.sha1()
                written = 0
                with tempfile.NamedTemporaryFile(prefix='.nymeria_', suffix='.partial', dir=path.parent, delete=False) as f:
                    partial = Path(f.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        written += len(chunk)
                        if written > size:
                            raise ValueError('download larger than manifest size')
                        h.update(chunk)
                        f.write(chunk)
                if written != size or h.hexdigest() != expected:
                    raise ValueError('size or SHA-1 mismatch')
                # Each sequence/group is unique; do not overwrite any pre-existing file.
                with path.open('xb') as dest, partial.open('rb') as src:
                    shutil.copyfileobj(src, dest, 1024 * 1024)
                partial.unlink()
                result['status'] = 'downloaded_verified'
                return result
        except Exception as exc:
            if partial is not None and partial.exists():
                partial.unlink()
            if path.exists():
                result['status'] = 'existing_after_failure'
                return result
            if attempt == 2:
                result['status'] = 'failed'
                result['error_type'] = type(exc).__name__
                if isinstance(exc, HTTPError):
                    result['http_status'] = exc.code
                return result
            time.sleep(1 + attempt * 2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--reserve-bytes', type=int, default=1073741824)
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or args.reserve_bytes < 0:
        p.error('workers must be between 1 and 8 and reserve must be non-negative')
    obj = json.loads(args.manifest.read_text(encoding='utf-8'))
    sequences = obj.get('sequences')
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError('manifest has no sequences')
    tasks = []
    for seq, groups in sequences.items():
        if not SAFE_SEQUENCE.fullmatch(seq) or not isinstance(groups, dict):
            raise ValueError('invalid sequence identifier or entry')
        for group in OUTPUTS:
            if group in groups:
                info = groups[group]
                if not isinstance(info, dict) or not all(k in info for k in ('sha1sum', 'file_size_bytes', 'download_url')):
                    raise ValueError(f'incomplete metadata for {group}')
                tasks.append((seq, group, info))
    args.output.mkdir(parents=True, exist_ok=True)
    need = sum(int(info['file_size_bytes']) for _, _, info in tasks)
    free = shutil.disk_usage(args.output).free
    if free - need * 2 < args.reserve_bytes:
        raise RuntimeError('insufficient free space for annotation files, temporary copies and reserve')
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, seq, group, info, args.output) for seq, group, info in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f'processed={len(results)}/{len(tasks)}', flush=True)
    counts = {}
    for result in results:
        counts[result['status']] = counts.get(result['status'], 0) + 1
    summary = {'selected_records': len(tasks), 'sequence_count': len(sequences),
               'selected_bytes': need, 'statuses': counts,
               'records': sorted(results, key=lambda x: (x['sequence'], x['group']))}
    report = args.output / 'download_summary.json'
    tmp_report = report.with_suffix('.json.partial')
    tmp_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp_report, report)
    print(json.dumps({'selected_records': len(tasks), 'statuses': counts, 'summary': str(report)}), flush=True)
    if any(status not in ('downloaded_verified', 'verified_existing') for status in counts):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
