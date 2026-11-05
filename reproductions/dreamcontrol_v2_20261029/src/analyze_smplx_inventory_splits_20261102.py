"""Report nominal source-group and interval overlap, without training acceptance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--dataset', choices=['BMLmovi', 'Eyes_Japan_Dataset'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = [a.mapping, a.inventory/'records.jsonl', a.inventory/'summary.json']
    hashes = {str(p): sha(p) for p in paths}
    summary = json.loads(paths[2].read_text(encoding='utf-8'))
    if summary.get('event') != 'complete' or summary.get('dataset') != a.dataset:
        raise ValueError('a completed matching inventory is required')
    original = {}
    for line in a.mapping.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['motion_id'] in original:
            raise ValueError('duplicate HumanML ID')
        original[r['motion_id']] = r
    groups, fps = [], Counter()
    samples = Counter()
    matched_ids = set()
    for line in paths[1].read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('duplicate_key') or r.get('training_accepted') is not False:
            raise ValueError('duplicate source key or unexpected acceptance')
        rows = []
        for candidate in r['nominal_rows']:
            sid = candidate['motion_id']
            if sid in matched_ids or sid not in original:
                raise ValueError('duplicate or missing ID')
            matched_ids.add(sid)
            origin = original[sid]
            if (candidate['path'] != origin['source_path'] or str(candidate['start']) != str(origin['index_start_20'])
                or str(candidate['end']) != str(origin['index_end_20'])):
                raise ValueError('mapping mismatch')
            split = origin['official_split']
            if split not in ('train', 'val', 'test'):
                raise ValueError('unexpected official split')
            samples[split] += 1
            rows.append({'id': sid, 'split': split, 'mirrored': origin['mirrored'],
                         'start': int(candidate['start']), 'end': int(candidate['end'])})
        if 'error' not in r:
            fps[str(r['declared_fps'])] += 1
        splits = sorted({x['split'] for x in rows})
        pairs = []
        for x, y in itertools.combinations(rows, 2):
            if x['split'] == y['split'] or x['mirrored'] != y['mirrored']:
                continue
            lo, hi = max(x['start'], y['start']), min(x['end'], y['end'])
            if lo < hi:
                pairs.append({'ids': [x['id'], y['id']], 'splits': [x['split'], y['split']],
                              'nominal_shared_interval': [lo, hi], 'mirrored': x['mirrored']})
        if len(splits) > 1:
            groups.append({'member': r['member'], 'source_sha256': r.get('member_sha256'),
                           'splits': splits, 'nominal_same_transform_overlaps': pairs})
    if len(matched_ids) != summary['counts']['matched_id_rows']:
        raise ValueError('matched row count differs from inventory')
    if any(sha(Path(p)) != d for p, d in hashes.items()):
        raise RuntimeError('input changed')
    report = {'dataset': a.dataset, 'matched_ids': len(matched_ids), 'matched_ids_by_official_split': dict(samples),
              'declared_fps_counts_by_source_member': dict(fps), 'cross_split_nominal_source_groups': len(groups),
              'groups_with_same_transform_nominal_overlap': sum(bool(g['nominal_same_transform_overlaps']) for g in groups),
              'same_transform_nominal_overlap_pairs': sum(len(g['nominal_same_transform_overlaps']) for g in groups),
              'groups': groups, 'input_sha256': hashes, 'script_sha256': sha(Path(__file__)),
              'training_ready': False, 'limitations': ['Ranges and split labels are nominal HumanML index metadata.',
                  'Name matches do not establish recording identity, actual timing or byte-identical output.',
                  'Mirrors are distinguished in pair counts; subject and other cross-source duplication remain unresolved.']}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('dataset','matched_ids','cross_split_nominal_source_groups',
                                             'groups_with_same_transform_nominal_overlap','same_transform_nominal_overlap_pairs')}))


if __name__ == '__main__':
    main()
