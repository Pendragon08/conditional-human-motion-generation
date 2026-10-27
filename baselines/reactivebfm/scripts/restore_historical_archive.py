"""Verify archived pieces and restore into a NEW directory, without pickle loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


class Pieces:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None

    def read(self, n):
        if n < 0:
            raise ValueError('Unbounded reads unsupported')
        blocks = []
        remaining = n
        while remaining:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file = path.open('rb')
            block = self.file.read(remaining)
            if not block:
                self.file.close()
                self.file = None
                continue
            blocks.append(block)
            remaining -= len(block)
        return b''.join(blocks)

    def close(self):
        if self.file is not None:
            self.file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--parts-dir', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    index = json.loads(args.index.read_text(encoding='utf-8'))
    if args.destination.exists():
        raise ValueError('Destination must be a new directory; original paths are never overwritten')
    pieces = []
    full = hashlib.sha256()
    total = 0
    for row in index['pieces_in_order']:
        if Path(row['name']).name != row['name'] or '/' in row['name'] or '\\' in row['name']:
            raise ValueError('Invalid piece name')
        path = args.parts_dir / row['name']
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                h.update(block)
                full.update(block)
                total += len(block)
        if path.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
            raise ValueError('Piece SHA256/size mismatch')
        pieces.append(path)
    if total != index['archive_bytes'] or full.hexdigest() != index['archive_sha256']:
        raise ValueError('Whole archive SHA256/size mismatch')
    expected = {row['archive_relative_path']: row for row in index['files']}
    if len(expected) != len(index['files']):
        raise ValueError('Duplicate manifest entry')
    for name in expected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe archive entry')
    args.destination.mkdir(parents=True, exist_ok=False)
    destination = args.destination.resolve()
    seen = set()
    source = Pieces(pieces)
    try:
        with tarfile.open(fileobj=source, mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected/nonregular archive entry')
                row = expected[member.name]
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                if not target.resolve().is_relative_to(destination) or target.exists() or member.size != row['bytes']:
                    raise ValueError('Unsafe/existing target or member size mismatch')
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.partial')
                h = hashlib.sha256()
                with tar.extractfile(member) as f, partial.open('xb') as out:
                    for block in iter(lambda: f.read(1048576), b''):
                        out.write(block)
                        h.update(block)
                if partial.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
                    raise ValueError('Restored member differs; partial evidence retained')
                os.replace(partial, target)
                seen.add(member.name)
    finally:
        source.close()
    if seen != set(expected):
        raise ValueError('Archive missing members')
    receipt = {'status': 'RESTORED_SOURCE_BYTES_SHA256_VERIFIED', 'files': len(seen),
               'archive_sha256': index['archive_sha256'], 'destination': str(destination)}
    (destination / '_RESTORE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
