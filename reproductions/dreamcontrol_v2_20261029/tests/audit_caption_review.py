"""Independently verify review-bundle identity, sampled timestamps and PNG files."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--state-manifest', required=True, type=Path)
    ap.add_argument('--state-root', required=True, type=Path)
    ap.add_argument('--urdf', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    report = json.loads((args.bundle / 'index.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    states = json.loads(args.state_manifest.read_text())
    assert report['manifest_sha256'] == sha(args.manifest) == states['preview_manifest_sha256']
    assert report['state_manifest_sha256'] == sha(args.state_manifest)
    assert report['urdf_sha256'] == sha(args.urdf)
    original = {r['sample_id']: r for r in manifest['records']}
    by_id = {r['sample_id']: r for r in states['records']}
    reviewed = {r['sample_id']: r for r in report['cases']}
    assert len(original) == len(manifest['records']) == len(by_id) == len(reviewed)
    assert set(original) == set(by_id) == set(reviewed)
    for name, row in reviewed.items():
        source, state = original[name], by_id[name]
        assert row['caption'] == source['caption'] and row['split'] == source['split']
        assert row['feature_sha256'] == source['feature_sha256'] == state['feature_sha256']
        path = args.state_root / state['state_path']
        assert row['state_sha256'] == state['state_sha256'] == sha(path)
        with np.load(path, allow_pickle=False) as data:
            times = data['sample_times']
        ids = row['frame_indices']
        assert len(ids) == 6 and ids[0] == 0 and ids[-1] == len(times) - 1
        assert all(a < b for a, b in zip(ids, ids[1:]))
        assert np.array_equal(times[ids], np.asarray(row['sample_times']))
        image = args.bundle / row['image']
        header = image.read_bytes()[:24]
        assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
        width, height = struct.unpack('>II', header[16:24])
        assert width >= 1000 and height >= 400 and sha(image) == row['image_sha256']
    result = {'index_sha256': sha(args.bundle / 'index.json'), 'case_count': len(reviewed),
              'identity_timeline_png_checks_passed': True, 'training_ready': False,
              'limitations': ['Images and FK not independently recomputed.',
                              'Does not determine caption semantics or motion physical quality.']}
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
