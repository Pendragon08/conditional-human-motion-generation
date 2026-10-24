"""Select and verify the independently patched copy; never mutate upstream."""
from pathlib import Path
import hashlib
import json
import sys


def activate():
    root = Path(__file__).resolve().parents[1]
    copy = root/'cache/guarded_upstream'
    manifest = json.loads((copy/'INDEPENDENT_REPAIRS.json').read_text())
    for name, hashes in manifest['sources'].items():
        if hashlib.sha256((copy/name).read_bytes()).hexdigest() != hashes['patched_sha256']:
            raise ValueError('Repaired source hash mismatch: '+name)
        if hashlib.sha256((root/'upstream'/name).read_bytes()).hexdigest() != hashes['original_sha256']:
            raise ValueError('Original source hash mismatch: '+name)
    sys.path.insert(0, str(copy/'source/whole_body_tracking'))
    print('PHYGILE_INDEPENDENT_REPAIRS_ACTIVE', json.dumps(manifest['sources']), flush=True)
    return manifest


if __name__ == '__main__':
    import runpy
    activate()
    train = Path(__file__).resolve().parents[1]/'cache/guarded_upstream/scripts/rsl_rl/train.py'
    sys.path.insert(0, str(train.parent))
    sys.argv[0] = str(train)
    runpy.run_path(str(train), run_name='__main__')
