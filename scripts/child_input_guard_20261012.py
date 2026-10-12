"""Prospective loaded-input binding; does not certify historical experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def loaded_motion_hash(arrays):
    # Same canonical content contract as physics_guided_com_lineage_v24.
    h = hashlib.sha256()
    for key in sorted(arrays.files):
        value = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def verify(root, versions_path, request, loaded_replay):
    root, versions_path = Path(root), Path(versions_path)
    if sha(versions_path) != request['versions']:
        raise ValueError('Versions file differs from bound request')
    versions = json.loads(versions_path.read_text(encoding='utf8'))
    for name, digest in versions['files'].items():
        path = (root / name).resolve()
        # Version manifest must not silently bind unrelated paths.
        try:
            path.relative_to(root.resolve())
        except ValueError:
            raise ValueError('Version manifest path outside project: ' + name)
        if sha(path) != digest:
            raise ValueError('Version-bound source changed: ' + name)
    if request['mode'] == 'replay':
        if loaded_replay is None or loaded_motion_hash(loaded_replay) != request['replay_motion']:
            raise ValueError('Actual loaded replay content differs from request')
    elif loaded_replay is not None or 'replay_motion' in request:
        raise ValueError('Replay reference supplied to a generation request')
    return dict(versions_sha256=request['versions'],
                verified_source_files=len(versions['files']),
                loaded_replay_content_sha256=(loaded_motion_hash(loaded_replay)
                    if loaded_replay is not None else None),
                scope='loaded replay and project source binding only; no upstream/weight/state equality claim',
                historical_certification=False)


def check():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'example.py'
        source.write_text('x = 1\n', encoding='utf8')
        versions = root / 'versions.json'
        versions.write_text(json.dumps({'files': {'example.py': sha(source)}}), encoding='utf8')
        motion = root / 'motion.npz'
        np.savez_compressed(motion, reference=np.arange(18, dtype=np.float32).reshape(2, 3, 3))
        with np.load(motion, allow_pickle=False) as archive:
            request = dict(mode='replay', versions=sha(versions), replay_motion=loaded_motion_hash(archive))
            verify(root, versions, request, archive)
            def reject(call):
                try:
                    call()
                except ValueError:
                    return
                raise AssertionError('Invalid input accepted')
            reject(lambda: verify(root, versions, dict(request, replay_motion='0' * 64), archive))
            reject(lambda: verify(root, versions, dict(request, versions='0' * 64), archive))
            reject(lambda: verify(root, versions, request, None))
            reject(lambda: verify(root, versions, dict(request, mode='native'), archive))
            source.write_text('x = 2\n', encoding='utf8')
            reject(lambda: verify(root, versions, request, archive))
        source.write_text('x = 1\n', encoding='utf8')
        with np.load(motion, allow_pickle=False) as archive:
            # Hash the loaded data, not a possibly changed pathname after load.
            original_hash = loaded_motion_hash(archive)
        np.savez_compressed(motion, reference=np.zeros((2, 3, 3), dtype=np.float32))
        with np.load(motion, allow_pickle=False) as changed:
            reject(lambda: verify(root, versions, dict(request, replay_motion=original_hash), changed))
    print('CHILD_INPUT_BINDING_FIXTURES_PASS')


if __name__ == '__main__':
    check()
