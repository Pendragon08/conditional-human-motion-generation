"""Report nested NPY sizes without deserializing GRAB object fields."""
import argparse
import hashlib
import io
import json
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('members', nargs='+')
    args = parser.parse_args()
    h = hashlib.sha256()
    with open(args.archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    results = []
    with zipfile.ZipFile(args.archive) as archive:
        for name in args.members:
            info = archive.getinfo(name)
            if info.file_size > 128 * 1024 * 1024:
                raise ValueError('outer member exceeds the safety bound')
            with zipfile.ZipFile(io.BytesIO(archive.read(info))) as nested:
                results.append({'member': name, 'outer_bytes': info.file_size,
                                'fields': {i.filename: {'bytes': i.file_size, 'compressed_bytes': i.compress_size}
                                           for i in nested.infolist()}})
    print(json.dumps({'archive_sha256': h.hexdigest(), 'members': results}, indent=2))


if __name__ == '__main__':
    main()
