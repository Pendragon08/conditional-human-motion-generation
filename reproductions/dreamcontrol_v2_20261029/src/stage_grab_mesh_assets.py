"""Stage verified GRAB mesh archives in an isolated official-style layout."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

BASE = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE = BASE / 'data/GRAB'
DEST = BASE / 'grab_mesh_stage_20261030'
FILES = {
    'tools__subject_meshes__male.zip': ('tools/subject_meshes', '897a4991ff8fd63a118b6d5ea3d052b7a7c922dbb83b5be1abd29bb0242ec62d'),
    'tools__subject_meshes__female.zip': ('tools/subject_meshes', '9a2c0ec910500158a5a6d9bf75042fd22acc35c0abcf5df094c463d8fc4756f7'),
    'tools__object_meshes__contact_meshes.zip': ('tools/object_meshes', '8f71d5d131899d5b7684f35bcf7b64d499435a9bb68cde95b54fc19279a7fe64'),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    inventory = []
    for name, (prefix, expected) in FILES.items():
        path = SOURCE / name
        if digest(path) != expected:
            raise ValueError('source archive hash mismatch: ' + name)
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError('source archive CRC failed: ' + name)
            for info in z.infolist():
                p = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(mode):
                    raise ValueError('unsafe archive member: ' + info.filename)
                if not p.parts or p.parts[0] not in ('male', 'female', 'contact_meshes'):
                    raise ValueError('unexpected asset root: ' + info.filename)
                if info.file_size > 100 * 1024 * 1024:
                    raise ValueError('oversized asset')
                inventory.append((name, prefix, info.filename, info.is_dir(), info.file_size))
    files = [str(PurePosixPath(prefix) / member) for _, prefix, member, isdir, _ in inventory if not isdir]
    if len(files) != len(set(files)) or sum(i[4] for i in inventory) > 1024 ** 3:
        raise ValueError('duplicate destinations or excessive size')
    DEST.mkdir(mode=0o700)
    written = []
    for name, (prefix, _) in FILES.items():
        with zipfile.ZipFile(SOURCE / name) as z:
            for info in z.infolist():
                target = DEST / prefix / PurePosixPath(info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, target.open('xb') as dst:
                        for chunk in iter(lambda: src.read(1024 * 1024), b''):
                            dst.write(chunk)
                    written.append({'path': str(target.relative_to(DEST)), 'bytes': target.stat().st_size, 'sha256': digest(target)})
    required = ['tools/subject_meshes/male/s2.ply', 'tools/object_meshes/contact_meshes/airplane.ply']
    if any(not (DEST / p).is_file() for p in required):
        raise ValueError('missing Subject 2 witness assets')
    report = {'archive_sha256': {k: v[1] for k, v in FILES.items()}, 'required_witness_files': required,
              'files': written, 'warning': 'Staging only; individualized reconstruction, mesh/contact correspondence and physics are unverified.'}
    with (DEST / 'manifest.json').open('x', encoding='utf8') as out:
        json.dump(report, out, indent=2)
    print(json.dumps({'files': len(written), 'bytes': sum(x['bytes'] for x in written), 'required': required}))


if __name__ == '__main__':
    main()
