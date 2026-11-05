"""Report shared source-directory provenance across preview splits.

Directory identity is a structural warning, not proof of subject identity or
data leakage. This is not a training acceptance check and changes no split.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def summarize(manifest_bytes):
    manifest = json.loads(manifest_bytes)
    records = manifest['records']
    groups = {}
    seen = set()
    for record in records:
        sample, split, source = (record[k] for k in ('sample_id', 'split', 'source_id'))
        if not all(isinstance(v, str) and v for v in (sample, split, source)):
            raise ValueError('invalid source record')
        if sample in seen or split not in ('train', 'val', 'test'):
            raise ValueError('duplicate sample or invalid split')
        seen.add(sample)
        path = PurePosixPath(source)
        if path.is_absolute() or '..' in path.parts or len(path.parts) < 3:
            raise ValueError('unsafe or incomplete source path')
        parent = str(path.parent)
        groups.setdefault(parent, []).append({'sample_id': sample, 'split': split})
    crossing = {key: rows for key, rows in groups.items()
                if len({row['split'] for row in rows}) > 1}
    return {
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'samples': len(records), 'source_parent_directories': len(groups),
        'source_parent_directories_crossing_splits': crossing,
        'training_ready': False,
        'limitations': [
            'Directory grouping is a conservative structural view, not verified subject identity.',
            'Related recordings may also occur in differently named directories.',
            'The original manifest and official splits were not changed.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.manifest.read_bytes())
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
