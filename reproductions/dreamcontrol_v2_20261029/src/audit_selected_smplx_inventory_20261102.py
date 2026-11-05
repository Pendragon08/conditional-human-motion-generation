"""Independently cross-check a completed read-only SMPL-X inventory.

This re-parses mapping and inventory JSON, not the compressed archive contents.
Nominal names and file-header frame rates are not recording identity or clock truth.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key(subject, name):
    return (subject.casefold(), re.sub(r'[^a-z0-9]', '', name.casefold()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--dataset', choices=['BMLmovi', 'Eyes_Japan_Dataset'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inputs = {str(x): sha(x) for x in (a.mapping, a.inventory/'records.jsonl', a.inventory/'summary.json', a.inventory/'progress.jsonl')}
    summary = json.loads((a.inventory/'summary.json').read_text(encoding='utf-8'))
    progress = [json.loads(x) for x in (a.inventory/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    if (not progress or progress[0]['event'] != 'start' or progress[-1]['event'] != 'complete'
        or summary['event'] != 'complete' or summary['dataset'] != a.dataset
        or progress[0]['mapping_sha256'] != inputs[str(a.mapping)]):
        raise ValueError('incomplete or mismatched inventory')
    st = a.archive.stat()
    if st.st_size != progress[0]['archive_size'] or st.st_mtime_ns != progress[0]['archive_mtime_ns']:
        raise ValueError('archive metadata changed')
    mapping = defaultdict(list)
    for line in a.mapping.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        path = r['source_path'].replace('\\', '/')
        parts = path.split('/')
        if len(parts) >= 4 and parts[-3] == a.dataset and parts[-1].endswith('_poses.npy'):
            mapping[key(parts[-2], parts[-1][:-len('_poses.npy')])].append(
                (r['motion_id'], path, str(r['index_start_20']), str(r['index_end_20'])))
    counts = Counter()
    seen = set()
    duplicate_keys = []
    for line in (a.inventory/'records.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        parts = row['member'].replace('\\', '/').split('/')
        if len(parts) != 3 or parts[0] != a.dataset or not parts[2].endswith('_stageii.npz'):
            raise ValueError('unexpected tar member name')
        k = key(parts[1], parts[2][:-len('_stageii.npz')])
        if list(k) != row['name_key'] or k not in mapping or row['training_accepted'] is not False:
            raise ValueError('key or acceptance mismatch')
        if row['duplicate_key'] != (k in seen):
            raise ValueError('duplicate flag mismatch')
        if k in seen:
            duplicate_keys.append(list(k))
        seen.add(k)
        tuples = [(r['motion_id'], r['path'], str(r['start']), str(r['end'])) for r in row['nominal_rows']]
        if Counter(tuples) != Counter(mapping[k]):
            raise ValueError('nominal mapping mismatch')
        counts['matched_members'] += 1
        counts['matched_id_rows'] += len(tuples)
        if 'error' in row:
            counts['member_errors'] += 1
            counts['unresolved_id_rows'] += len(tuples)
            continue
        fps, shape = row['declared_fps'], row['poses_shape']
        if not math.isfinite(fps) or fps < 20 or len(shape) != 2 or len(row['member_sha256']) != 64:
            raise ValueError('invalid reported source metadata')
        stride = int(fps/20)
        n = (shape[0]+stride-1)//stride
        if stride != row['integer_stride'] or n != row['nominal_20hz_count']:
            raise ValueError('nominal sample count mismatch')
        for r in row['nominal_rows']:
            try:
                start, end = int(r['start']), int(r['end'])
                if start < 0 or end <= start:
                    raise ValueError('invalid interval')
                inside = end <= n
                if r.get('interval_within_nominal_count') is not inside:
                    raise ValueError('range indicator mismatch')
                counts['covered_id_rows' if inside else 'short_id_rows'] += 1
            except ValueError as e:
                if str(e) == 'range indicator mismatch':
                    raise
                if r.get('interval_within_nominal_count') is not None:
                    raise ValueError('unresolved range mismatch')
                counts['unresolved_id_rows'] += 1
    for label in ('matched_members', 'matched_id_rows', 'covered_id_rows', 'short_id_rows', 'unresolved_id_rows', 'member_errors'):
        if counts[label] != summary['counts'][label]:
            raise ValueError('summary counter mismatch: '+label)
    if len(mapping) != summary['expected_nominal_keys'] or len(seen) != summary['matched_unique_keys']:
        raise ValueError('summary unique-key count mismatch')
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise RuntimeError('audit input changed')
    result = {'dataset': a.dataset, 'counts': dict(counts), 'expected_nominal_keys': len(mapping),
              'matched_unique_keys': len(seen), 'unmatched_nominal_keys': len(set(mapping)-seen),
              'duplicate_archive_keys': duplicate_keys, 'input_sha256': inputs,
              'archive_size': st.st_size, 'archive_mtime_ns': st.st_mtime_ns,
              'training_ready': False, 'script_sha256': sha(Path(__file__)),
              'limitations': ['Recomputes from inventory and mapping; does not re-read archive members.',
                              'Names and header metadata do not prove recording identity, actual clock, semantics or license.']}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'dataset': a.dataset, **result['counts'],
                      'matched_unique_keys': len(seen), 'expected_nominal_keys': len(mapping)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
