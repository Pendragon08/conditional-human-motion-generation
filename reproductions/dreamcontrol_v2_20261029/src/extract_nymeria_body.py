"""Extract only body/xdata.npz from a signed Nymeria archive via byte ranges.

The extracted ZIP member is verified using the ZIP central-directory CRC-32;
the enclosing archive SHA-1 is not verifiable without downloading all of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

from nymeria_remote_zip import RemoteZipReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reserve-bytes', type=int, default=1073741824)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.sequence):
        raise ValueError('invalid sequence identifier')
    data = json.loads(args.manifest.read_text(encoding='utf-8'))
    info = data['sequences'][args.sequence]['body_motion']
    stream = RemoteZipReader(info['download_url'], int(info['file_size_bytes']), block_size=8 * 1024 * 1024)
    outdir = args.output_root / args.sequence / 'body'
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / 'xdata.npz'
    if destination.exists():
        raise FileExistsError('destination already exists; preserving it')
    with zipfile.ZipFile(stream) as z:
        item = z.getinfo('body/xdata.npz')
        if shutil.disk_usage(outdir).free < item.file_size + args.reserve_bytes:
            raise RuntimeError('insufficient staging space plus reserve')
        temp = None
        try:
            h = hashlib.sha256()
            with tempfile.NamedTemporaryFile(prefix='.body_', suffix='.partial', dir=outdir, delete=False) as f:
                temp = Path(f.name)
                with z.open(item) as member:
                    for chunk in iter(lambda: member.read(1024 * 1024), b''):
                        h.update(chunk)
                        f.write(chunk)
            if temp.stat().st_size != item.file_size:
                raise ValueError('member size mismatch')
            # Hard-link is atomic and fails when the destination exists.
            import os
            os.link(temp, destination)
            temp.unlink()
        finally:
            if temp is not None and temp.exists():
                temp.unlink()
    result = {'sequence': args.sequence, 'file': str(destination), 'bytes': item.file_size,
              'sha256': h.hexdigest(), 'zip_crc32': f'{item.CRC:08x}',
              'range_bytes': stream.bytes_transferred, 'range_requests': stream.requests,
              'verification': 'zip member CRC-32 and extracted SHA-256; archive SHA-1 not verified'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # urllib exceptions can include signed credentials; do not print them.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)
