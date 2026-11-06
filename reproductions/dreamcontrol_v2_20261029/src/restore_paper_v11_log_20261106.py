"""Restore one archived paper_v11 log, keeping the archive and refusing overwrites."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

BANK = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105/new_references_paper_v11')
FLOOR = 8 * 1024**3 + 456 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('marker', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    marker = args.marker.resolve(strict=True)
    entry = json.loads(marker.read_text())
    source, archive = Path(entry['source']), Path(entry['archive'])
    assert source.parent.parent.parent == BANK and source.parent.resolve() == source.parent
    assert source.name in ('sample.log', 'postopt.log', 'native.log')
    assert marker == source.with_name(source.name + '.archived.json')
    assert archive == source.with_name(source.name + '.gz') and archive.resolve(strict=True) == archive
    assert digest(archive) == entry['gzip_sha256'] and archive.stat().st_size == entry['gzip_bytes']
    if source.exists() or source.is_symlink():
        assert not source.is_symlink() and digest(source) == entry['source_sha256'], 'Different original content: refusing overwrite'
        print(json.dumps({'status': 'already_restored_identical', 'source': str(source)}))
        return
    if not args.execute:
        print(json.dumps({'status': 'archive_verified_restore_ready', 'source': str(source), 'bytes': entry['source_bytes']}))
        return
    assert shutil.disk_usage(source.parent).free >= FLOOR + entry['source_bytes'] + 20 * 1024**2
    temp = source.with_name(source.name + '.restore_pending')
    h, size = hashlib.sha256(), 0
    with temp.open('xb') as output:
        with gzip.open(archive, 'rb') as decoded:
            for chunk in iter(lambda: decoded.read(1024**2), b''):
                h.update(chunk)
                size += len(chunk)
                output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    assert h.hexdigest() == entry['source_sha256'] and size == entry['source_bytes']
    assert digest(archive) == entry['gzip_sha256']
    os.chmod(temp, int(entry['source_identity']['st_mode']) & 0o777)
    mtime = int(entry['source_identity']['st_mtime_ns'])
    os.utime(temp, ns=(mtime, mtime))
    os.link(temp, source, follow_symlinks=False)
    os.unlink(temp)
    print(json.dumps({'status': 'restored_verified_archive_preserved', 'source': str(source), 'sha256': h.hexdigest()}))


if __name__ == '__main__':
    main()
