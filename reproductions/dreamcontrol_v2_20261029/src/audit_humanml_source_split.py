"""Audit shared source paths across HumanML split IDs; no motion is decoded."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE = ROOT / 'humanml_raw_mapping_v2_20261030' / 'mapping.jsonl'
OUT = Path(__file__).resolve().parent
EXPECTED = '358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    if sha(SOURCE) != EXPECTED: raise ValueError('mapping hash changed')
    rows = [json.loads(s) for s in SOURCE.read_text(encoding='utf8').splitlines()]
    if len(rows) != 29228 or len({r['motion_id'] for r in rows}) != len(rows):
        raise ValueError('unexpected mapping coverage')
    counts = Counter(r['official_split'] for r in rows)
    if counts != {'train':23384, 'val':1460, 'test':4384}: raise ValueError('split totals differ')
    groups = defaultdict(list)
    for row in rows:
        path = row['source_path']
        if not path or not path.startswith('./pose_data/'): raise ValueError('unexpected source path')
        groups[path].append({'motion_id':row['motion_id'], 'split':row['official_split'],
                             'mirrored':row['mirrored'], 'index_start_20':row['index_start_20'],
                             'index_end_20':row['index_end_20'], 'mapping_status':row['status']})
    overlaps = []
    combinations = Counter()
    by_dataset = defaultdict(Counter)
    for path, members in sorted(groups.items()):
        splits = sorted({m['split'] for m in members})
        if len(splits) < 2: continue
        combinations['+'.join(splits)] += 1
        dataset = path.split('/')[2]
        by_dataset[dataset]['source_paths'] += 1
        by_dataset[dataset]['ids'] += len(members)
        same_crop = defaultdict(list)
        for m in members:
            same_crop[(m['index_start_20'], m['index_end_20'])].append(m['split'])
        crop_cross_split = sum(1 for s in same_crop.values() if len(set(s)) > 1)
        overlaps.append({'source_path': path, 'dataset': dataset, 'splits': splits,
                         'same_declared_indices_across_splits': crop_cross_split,
                         'members': sorted(members, key=lambda x:x['motion_id'])})
    jsonl = OUT / 'cross_split_sources.jsonl'
    with jsonl.open('x', encoding='utf8', newline='\n') as f:
        for group in overlaps: f.write(json.dumps(group, sort_keys=True, allow_nan=False) + '\n')
    result = {'mapping_sha256':EXPECTED, 'script_sha256':sha(Path(__file__)),
              'cross_split_sources_sha256':sha(jsonl), 'official_id_counts':dict(counts),
              'unique_exact_source_paths':len(groups), 'cross_split_source_paths':len(overlaps),
              'cross_split_ids':sum(len(g['members']) for g in overlaps),
              'cross_split_source_paths_with_same_declared_indices':sum(g['same_declared_indices_across_splits'] > 0 for g in overlaps),
              'split_combinations':dict(combinations), 'by_dataset':{k:dict(v) for k,v in sorted(by_dataset.items())},
              'limitations':'Exact source paths and nominal declared index pairs only; mapping and candidate paths do not establish recording identity, actual timestamps, crop overlap or reconstructability. A shared source is a possible evaluation leakage, not proof of identical motions. No split was changed and no data was excluded.'}
    with (OUT / 'report.json').open('x', encoding='utf8') as f: json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == '__main__': main()
