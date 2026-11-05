"""Summarize the authorized Nymeria URL manifest without printing signed URLs.

The manifest may contain temporary download credentials. Do not commit it,
print it, or include it in public experiment artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = ('LICENSE', 'metadata_json', 'body_motion',
          'narration_motion_narration_csv', 'narration_atomic_action_csv',
          'narration_activity_summarization_csv')


def inspect(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    sequences = obj.get('sequences')
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError('manifest has no sequences')
    groups = {name: {'count': 0, 'size_bytes': 0, 'missing': 0} for name in GROUPS}
    invalid = 0
    max_bytes = 0
    for entry in sequences.values():
        if not isinstance(entry, dict):
            raise ValueError('invalid sequence entry')
        for name in GROUPS:
            obj = entry.get(name)
            if obj is None:
                groups[name]['missing'] += 1
                continue
            if not isinstance(obj, dict):
                invalid += 1
                continue
            size = obj.get('file_size_bytes')
            if not isinstance(size, int) or size < 0 or not obj.get('sha1sum') or not obj.get('download_url'):
                invalid += 1
                continue
            groups[name]['count'] += 1
            groups[name]['size_bytes'] += size
            max_bytes = max(max_bytes, size)
    return {'sequence_count': len(sequences), 'groups': groups,
            'invalid_selected_records': invalid,
            'compressed_download_bytes': sum(group['size_bytes'] for group in groups.values()),
            'largest_individual_download_bytes': max_bytes,
            'note': 'Compressed sizes only; extraction and temporary files need additional free space.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    args = p.parse_args()
    print(json.dumps(inspect(args.manifest), indent=2))


if __name__ == '__main__':
    main()
